//! One native compute schedule with conservative allocation-level dependencies.
use super::*;
use std::collections::HashMap;

#[derive(Default)]
struct Access {
    write: Option<usize>,
    read: Option<usize>,
}
pub(super) enum Operation {
    Compute {
        native: Arc<executor::NativeDispatch>,
    },
    Copy {
        source: GpuBuffer,
        output: GpuBuffer,
        source_offset: u64,
        output_offset: u64,
        bytes: u64,
    },
}
#[derive(Default)]
pub(super) struct Schedule {
    operations: Vec<(usize, Operation)>,
    accesses: HashMap<usize, Access>,
}
pub(super) fn identity(buffer: &GpuBuffer) -> usize {
    Arc::as_ptr(&buffer.reservation) as usize
}
impl Schedule {
    pub fn take_native_resources(&mut self) -> Vec<Arc<executor::NativeDispatch>> {
        std::mem::take(&mut self.operations)
            .into_iter()
            .filter_map(|(_, operation)| match operation {
                Operation::Compute { native } => Some(native),
                _ => None,
            })
            .collect()
    }
    pub fn push(&mut self, operation: Operation, buffers: &[(&GpuBuffer, bool)]) {
        let depth = self.record(
            buffers
                .iter()
                .map(|&(buffer, read)| (identity(buffer), read)),
        );
        self.operations.push((depth, operation));
    }
    pub fn push_prepared(&mut self, operation: Operation, hazards: &[(usize, bool)]) {
        let depth = self.record(hazards.iter().copied());
        self.operations.push((depth, operation));
    }
    fn record(&mut self, accesses: impl Iterator<Item = (usize, bool)> + Clone) -> usize {
        let depth = self.depth(accesses.clone());
        for (id, read_only) in accesses {
            let access = self.accesses.entry(id).or_default();
            if read_only {
                access.read = Some(access.read.map_or(depth, |old| old.max(depth)));
            } else {
                access.write = Some(depth);
                access.read = None;
            }
        }
        depth
    }
    fn depth(&self, accesses: impl Iterator<Item = (usize, bool)>) -> usize {
        accesses
            .filter_map(|(id, read_only)| {
                let access = self.accesses.get(&id)?;
                if read_only {
                    access.write
                } else {
                    access.write.into_iter().chain(access.read).max()
                }
            })
            .max()
            .map_or(0, |depth| depth + 1)
    }
    pub fn encode(
        &mut self,
        context: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        prefer_serial: bool,
    ) -> RuntimeResult<Vec<wgpu::CommandBuffer>> {
        for (_, operation) in &self.operations {
            if let Operation::Copy { source, output, .. } = operation
                && (!Arc::ptr_eq(&source.reservation.ledger, &context.ledger)
                    || !Arc::ptr_eq(&output.reservation.ledger, &context.ledger))
            {
                return Err(RuntimeError::with_kind(
                    ErrorKind::InvalidInput,
                    "native transfer buffers belong to another GPU context",
                ));
            }
        }
        let device = &context.device;
        sort_by_depth(&mut self.operations);
        let mut completed = Vec::new();
        let mut transfers_dirty = false;
        let mut index = 0;
        while index < self.operations.len() {
            if matches!(self.operations[index].1, Operation::Compute { .. }) {
                let mut dispatches = Vec::new();
                while let Some((depth, Operation::Compute { native })) = self.operations.get(index)
                {
                    dispatches.push((*depth, native.as_ref()));
                    index += 1;
                }
                // Vulkan's managed encoder tracks transfer/compute hazards.
                // Keep the entire batch in one command buffer, including copies;
                // only the raw Metal bridge requires separate encoders.
                if !executor::SEPARATE_TRANSFER_ENCODERS {
                    executor::encode(context, encoder, &dispatches, prefer_serial)?;
                    continue;
                }
                if transfers_dirty {
                    let preceding = std::mem::replace(
                        encoder,
                        device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("resident transfer continuation"),
                        }),
                    );
                    completed.push(preceding.finish());
                    transfers_dirty = false;
                }
                // Raw compute encoders never share a command encoder with the
                // resource/queue bridge's copy recording API.
                let mut raw = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("native dependency chunk"),
                });
                executor::encode(context, &mut raw, &dispatches, prefer_serial)?;
                completed.push(raw.finish());
                continue;
            }
            transfers_dirty = true;
            match &self.operations[index].1 {
                Operation::Copy {
                    source,
                    output,
                    source_offset,
                    output_offset,
                    bytes,
                } => encoder.copy_buffer_to_buffer(
                    source,
                    *source_offset,
                    output,
                    *output_offset,
                    *bytes,
                ),
                Operation::Compute { .. } => unreachable!(),
            }
            index += 1;
        }
        Ok(completed)
    }
}
fn sort_by_depth<T>(operations: &mut [(usize, T)]) {
    operations.sort_by_key(|(depth, _)| *depth);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prepared_allocation_hazards_preserve_alias_and_cross_graph_dependencies() {
        let mut schedule = Schedule::default();
        // Binding aliases repeat one allocation. A simultaneous read/write
        // must not create a self-edge, but every subsequent use must wait.
        let first = [(10, true), (10, false), (20, true)];
        assert_eq!(schedule.record(first.into_iter()), 0);
        assert_eq!(schedule.record([(30, true), (40, false)].into_iter()), 0);
        assert_eq!(schedule.record([(10, true), (50, false)].into_iter()), 1);
        assert_eq!(schedule.record([(10, false), (40, true)].into_iter()), 2);
        assert_eq!(schedule.record([(50, true), (10, true)].into_iter()), 3);
        // Shared immutable weights create no extra dependency between graphs.
        assert_eq!(schedule.record([(20, true), (60, false)].into_iter()), 0);
    }

    #[test]
    fn read_write_dependencies_preserve_reused_storage() {
        let mut schedule = Schedule::default();
        // Two disjoint graphs may share a depth; a read follows its producer,
        // and reuse follows every preceding reader and writer.
        assert_eq!(schedule.record([(1, false)].into_iter()), 0);
        assert_eq!(schedule.record([(1, true)].into_iter()), 1);
        assert_eq!(schedule.record([(2, false)].into_iter()), 0);
        assert_eq!(schedule.record([(2, true)].into_iter()), 1);
        assert_eq!(schedule.record([(1, false)].into_iter()), 2);
        assert_eq!(schedule.record([(1, true)].into_iter()), 3);
        assert_eq!(schedule.record([(1, true), (1, false)].into_iter()), 4);
        assert_eq!(schedule.record([(1, true), (2, false)].into_iter()), 5);
    }

    #[test]
    fn dependency_depth_orders_every_conflicting_access() {
        let mut schedule = Schedule::default();
        let operations = (0..192)
            .map(|i| [(i % 11, true), ((i * 7 + 3) % 11, false)])
            .collect::<Vec<_>>();
        let depths = operations
            .iter()
            .map(|accesses| schedule.record(accesses.iter().copied()))
            .collect::<Vec<_>>();
        for later in 0..operations.len() {
            for earlier in 0..later {
                if operations[earlier].iter().any(|&(first, read_first)| {
                    operations[later].iter().any(|&(second, read_second)| {
                        first == second && !(read_first && read_second)
                    })
                }) {
                    assert!(depths[earlier] < depths[later], "{earlier} -> {later}");
                }
            }
        }
    }

    #[test]
    fn sorting_preserves_dependency_levels_and_equal_depth_chronology() {
        let mut schedule = Schedule::default();
        let accesses = [
            vec![(1, false)],           // Graph A producer.
            vec![(1, true)],            // Graph A consumer.
            vec![(2, false)],           // Independent graph B producer.
            vec![(2, true)],            // Graph B consumer.
            vec![(1, false)],           // Reuse A only after its reader.
            vec![(3, false)],           // Independent clear/write.
            vec![(1, true), (2, true)], // Join both graphs.
            vec![(2, false)],           // Reuse B only after the join.
        ];
        let mut operations: Vec<_> = accesses
            .iter()
            .enumerate()
            .map(|(index, accesses)| (schedule.record(accesses.iter().copied()), index))
            .collect();
        sort_by_depth(&mut operations);
        assert_eq!(
            operations,
            [
                (0, 0),
                (0, 2),
                (0, 5),
                (1, 1),
                (1, 3),
                (2, 4),
                (3, 6),
                (4, 7)
            ]
        );
    }
}
