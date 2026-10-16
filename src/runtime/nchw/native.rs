//! Outline whole kernel families without losing the ISA of their reductions.
//! Separate family modules also give rustc independent codegen partitions.
//!
//! The inline const selects one entry during monomorphization. A runtime `if`
//! here would make rustc collect every ISA entry for every geometry/store type.
macro_rules! entry {
    (
        $name:ident $(, $feature:literal)?;
        $implementation:ident <
            const $lanes:ident: usize, const $fused:ident: bool,
            $(const $parameter:ident: $kind:ty,)*
            ; $store_type:ident
        > ($($argument:ident: $argument_type:ty,)* ; $store:ident)
    ) => {
        $(#[target_feature(enable = $feature)])?
        #[inline(never)]
        #[allow(clippy::too_many_arguments)]
        unsafe fn $name<
            const $lanes: usize, const $fused: bool,
            $(const $parameter: $kind,)*
            $store_type: $crate::runtime::simd::StoreTransform,
        >($($argument: $argument_type,)* $store: &$store_type) {
            $implementation::<$lanes, $fused $(, $parameter)*>($($argument,)* $store);
        }
    };
}

macro_rules! outline {
    (
        $(#[$attribute:meta])*
        $visibility:vis fn $name:ident => $implementation:ident <
            const $lanes:ident: usize, const $fused:ident: bool
            $(, const $parameter:ident: $kind:ty)* $(,)?
        > ($($argument:ident: $argument_type:ty),* $(,)?)
        $store:ident: $store_type:ident
    ) => {
        $(#[$attribute])*
        #[inline(always)]
        #[allow(clippy::too_many_arguments)]
        $visibility fn $name<
            const $lanes: usize, const $fused: bool
            $(, const $parameter: $kind)*
        >($($argument: $argument_type,)* $store: &impl $crate::runtime::simd::StoreTransform) {
            $name::dispatch::<$lanes, $fused $(, $parameter)*, _>($($argument,)* $store);
        }

        $(#[$attribute])*
        mod $name {
            use super::*;

            #[inline(always)]
            #[allow(clippy::too_many_arguments)]
            pub(super) fn dispatch<
                const $lanes: usize, const $fused: bool,
                $(const $parameter: $kind,)*
                $store_type: $crate::runtime::simd::StoreTransform,
            >($($argument: $argument_type,)* $store: &$store_type) {
                $crate::runtime::nchw::native::entry!(portable;
                    $implementation<const $lanes: usize, const $fused: bool,
                        $(const $parameter: $kind,)* ; $store_type>
                    ($($argument: $argument_type,)* ; $store)
                );
                #[cfg(target_arch = "x86_64")]
                $crate::runtime::nchw::native::entry!(avx, "avx";
                    $implementation<const $lanes: usize, const $fused: bool,
                        $(const $parameter: $kind,)* ; $store_type>
                    ($($argument: $argument_type,)* ; $store)
                );
                #[cfg(target_arch = "x86_64")]
                $crate::runtime::nchw::native::entry!(avx_fma, "avx,fma";
                    $implementation<const $lanes: usize, const $fused: bool,
                        $(const $parameter: $kind,)* ; $store_type>
                    ($($argument: $argument_type,)* ; $store)
                );
                #[cfg(target_arch = "x86_64")]
                $crate::runtime::nchw::native::entry!(avx512, "avx512f,avx2,fma";
                    $implementation<const $lanes: usize, const $fused: bool,
                        $(const $parameter: $kind,)* ; $store_type>
                    ($($argument: $argument_type,)* ; $store)
                );

                let kernel: unsafe fn($($argument_type,)* &$store_type) = const {
                    assert!(matches!(($lanes, $fused), (4, false) | (8, false) | (8, true) | (16, true)));
                    #[cfg(target_arch = "x86_64")]
                    {
                        if $lanes == 16 && $fused {
                            avx512::<$lanes, $fused $(, $parameter)*, $store_type>
                        } else if $fused {
                            // AVX/FMA and AVX2/FMA share these float kernels.
                            avx_fma::<$lanes, $fused $(, $parameter)*, $store_type>
                        } else if $lanes == 8 {
                            avx::<$lanes, $fused $(, $parameter)*, $store_type>
                        } else {
                            portable::<$lanes, $fused $(, $parameter)*, $store_type>
                        }
                    }
                    #[cfg(not(target_arch = "x86_64"))]
                    { portable::<$lanes, $fused $(, $parameter)*, $store_type> }
                };
                // SAFETY: these private kernels are reached from NCHW's guarded
                // backend entries. Portable4 uses the baseline entry; native
                // lane/FMA combinations retain that backend's required ISA.
                unsafe { kernel($($argument,)* $store) }
            }
        }
    };
}

pub(super) use entry;
pub(super) use outline;
