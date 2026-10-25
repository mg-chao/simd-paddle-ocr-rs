void run(uint3 id, uint3 group, uint lane, threadgroup float *scratch) const {
  uint index = id.x + id.y * p(1);
  if (index >= p(0))
    return;
  uint op = p(2);
  if (op == 26) {
    if (index >= p(6))
      return;
    uint h = p(3), w = p(4), row = index / p(5), x = (index % p(5)) * 4,
         y = row % h, base = row * w + x;
    for (uint component = 0; component < 4 && x + component < w; component++) {
      uint element = base + component;
      float v = load_a(element);
      if (x + component + 1 < w)
        v = max(v, load_a(element + 1));
      if (y + 1 < h) {
        v = max(v, load_a(element + w));
        if (x + component + 1 < w)
          v = max(v, load_a(element + w + 1));
      }
      store_output(element, v);
    }
    return;
  }
  if (op == 25) {
    for (uint component = 0; component < 4; component++) {
      uint element = index * 4 + component;
      if (element >= p(0))
        break;
      float v = load_a(element), result;
      if (p(3) == 1) {
        float scaled = output_round(v / load_b(0));
        float error = output_round(error_function(scaled));
        float shifted = output_round(error + 1.0f);
        result = output_round(shifted * v) * 0.5f;
      } else if (p(3) == 2)
        result = v * output_round(clamp(v * 0.16666667f + 0.5f, 0.0f, 1.0f));
      else
        result = v * output_round(sigmoid(v));
      store_output(element, result);
    }
    return;
  }
  if (op == 24) {
    uint channels = p(3), plane = p(4), padded = p(5), channel = index % padded,
         pixel = (index / padded) % plane, batch = index / (padded * plane);
    store_output(index,
                 channel < channels
                     ? load_a((batch * channels + channel) * plane + pixel)
                     : 0.0f);
    return;
  }
  if (op == 22) {
    store_output(index + p(4), load_a(index + p(3)));
    return;
  }
  if (op == 23 || op == 17) {
    uint h = p(3), w = p(4), factor = p(5), oh = h * factor, ow = w * factor;
    float v = load_a(index / (oh * ow) * h * w +
                     ((index / ow) % oh / factor) * w + (index % ow) / factor);
    if (op == 23)
      v += load_b(index);
    store_output(index, v);
    return;
  }
  if (op <= 3) {
    uint remaining = index, ai = 0, bi = 0;
    for (uint axis = p(3); axis > 0;) {
      axis--;
      uint coordinate = remaining % p(4 + axis);
      remaining /= p(4 + axis);
      ai += coordinate * p(12 + axis);
      bi += coordinate * p(20 + axis);
    }
    float a = load_a(ai), b = load_b(bi);
    store_output(index, op == 0   ? a + b
                        : op == 1 ? a * b
                        : op == 2 ? a / b
                                  : a - b);
    return;
  }
  if (op >= 4 && op <= 9) {
    float v = load_a(index);
    if (op == 4)
      v *= v;
    else if (op == 5)
      v = sqrt(v);
    else if (op == 6)
      v = v > 0.0f ? v : 0.0f;
    else if (op == 7)
      v = error_function(v);
    else if (op == 8)
      v = sigmoid(v);
    else
      v = clamp(v * as_type<float>(p(3)) + as_type<float>(p(4)), 0.0f, 1.0f);
    store_output(index, v);
    return;
  }
  if (op == 11) {
    uint cin = p(3), cout = p(4), h = p(5), w = p(6), ow = w * 2, oh = h * 2,
         x = index % ow, y = (index / ow) % oh, co = (index / (ow * oh)) % cout,
         batch = index / (ow * oh * cout);
    float sum = p(7) ? load_c(co) : 0.0f;
    for (uint ci = 0; ci < cin; ci++)
      sum += load_a(((batch * cin + ci) * h + y / 2) * w + x / 2) *
             load_b((ci * cout + co) * 4 + (y % 2) * 2 + x % 2);
    store_output(index, epilogue(sum, index));
    return;
  }
  if (op == 12) {
    uint h = p(3), w = p(4), x = index % w, y = (index / w) % h;
    float v = load_a(index);
    if (x + 1 < w)
      v = max(v, load_a(index + 1));
    if (y + 1 < h) {
      v = max(v, load_a(index + w));
      if (x + 1 < w)
        v = max(v, load_a(index + w + 1));
    }
    store_output(index, v);
    return;
  }
  if (op == 13) {
    uint w = p(4), oh = p(5), ow = p(6), x = index % ow, y = (index / ow) % oh,
         base = index / (oh * ow) * p(3) * w;
    float sum = 0.0f;
    for (uint ky = 0; ky < 3; ky++)
      for (uint kx = 0; kx < 2; kx++)
        sum += load_a(base + (y * 3 + ky) * w + x * 2 + kx);
    store_output(index, sum / 6.0f);
    return;
  }
  if (op == 14) {
    uint channel = (index / p(4)) % p(3);
    float multiplier =
        load_b(channel) / sqrt(load_e(channel) + as_type<float>(p(5)));
    store_output(index, (load_a(index) - load_d(channel)) * multiplier +
                            load_c(channel));
    return;
  }
  if (op == 15) {
    uint remaining = index, source = p(4);
    for (uint axis = p(3); axis > 0;) {
      axis--;
      uint coordinate = remaining % p(8 + axis);
      remaining /= p(8 + axis);
      source += coordinate * p(16 + axis);
    }
    store_output(index, load_a(source));
    return;
  }
  if (op == 18) {
    uint rank = p(3), reduction = p(4), base = 0, input_stride = 1;
    for (uint axis = rank; axis > 0;) {
      axis--;
      uint dimension = p(8 + axis), stride = p(16 + axis);
      if (stride)
        base += ((index / stride) % dimension) * input_stride;
      input_stride *= dimension;
    }
    float sum = 0.0f;
    for (uint r = 0; r < reduction; r++) {
      uint remaining = r, offset = base, stride = 1;
      for (uint axis = rank; axis > 0;) {
        axis--;
        uint dimension = p(8 + axis);
        if (p(16 + axis) == 0) {
          offset += (remaining % dimension) * stride;
          remaining /= dimension;
        }
        stride *= dimension;
      }
      sum += load_a(offset);
    }
    store_output(index, sum / float(reduction));
    return;
  }
  if (op == 20) {
    uint rows = p(3), inner = p(4), columns = p(5),
         row = (index / columns) % rows, column = index % columns,
         batch = index / (rows * columns), ai = 0, bi = 0;
    for (uint axis = p(6); axis > 0;) {
      axis--;
      uint coordinate = batch % p(8 + axis);
      batch /= p(8 + axis);
      ai += coordinate * p(16 + axis);
      bi += coordinate * p(24 + axis);
    }
    float sum = 0.0f;
    for (uint k = 0; k < inner; k++)
      sum += load_a(ai + row * inner + k) * load_b(bi + k * columns + column);
    store_output(index, epilogue(sum, index));
    return;
  }
  if (op == 21) {
    uint columns = p(3), start = index * columns;
    float maximum = load_a(start);
    for (uint col = 1; col < columns; col++)
      maximum = max(maximum, load_a(start + col));
    float sum = 0.0f;
    for (uint col = 0; col < columns; col++) {
      float v = exp(load_a(start + col) - maximum);
      store_output(start + col, v);
      sum += v;
    }
    for (uint col = 0; col < columns; col++)
      store_output(start + col, float(output[start + col + p(60)]) / sum);
  }
}
