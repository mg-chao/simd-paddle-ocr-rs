// Explicit IEEE binary16 encoding retains sign and NaN classification even
// when a native float-to-half instruction canonicalizes NaNs.
inline half stored_half(float value) {
  uint bits = as_type<uint>(value), sign = (bits >> 16) & 0x8000u,
       magnitude = bits & 0x7fffffffu;
  if (magnitude >= 0x7f800000u)
    return as_type<half>(
        ushort(sign | 0x7c00u | ((magnitude & 0x7fffffu) ? 0x200u : 0u)));
  if (magnitude >= 0x477ff000u)
    return as_type<half>(ushort(sign | 0x7c00u));
  if (magnitude <= 0x33000000u)
    return as_type<half>(ushort(sign));
  if (magnitude < 0x38800000u) {
    uint shift = 126u - (magnitude >> 23u),
         mantissa = (magnitude & 0x7fffffu) | 0x800000u;
    uint rounded =
        (mantissa + ((1u << (shift - 1u)) - 1u) + ((mantissa >> shift) & 1u)) >>
        shift;
    return as_type<half>(ushort(sign | rounded));
  }
  uint rounded = magnitude + 0xfffu + ((magnitude >> 13u) & 1u);
  return as_type<half>(ushort(sign | ((rounded >> 13u) - 0x1c000u)));
}
