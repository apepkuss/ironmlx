// Per-row arithmetic copied from MLX `mlx/backend/metal/kernels/quantized.h`
// (load_vector, qdot and qmv_fast_impl), Copyright © 2023-2024 Apple Inc.,
// MIT License. Each output row is computed with exactly the MLX qmv_fast
// operation sequence; only the scheduling across rows that share an expert
// differs, so a row's value does not depend on how many rows are grouped.

template <int bits>
inline constexpr short grouped_pack_factor() {
  return (bits == 3 || bits == 5) ? 8 : (bits == 6 ? 4 : 32 / bits);
}

template <int bits>
inline constexpr short grouped_bytes_per_pack() {
  return ((bits & (bits - 1)) == 0) ? 4 : (bits == 5 ? 5 : 3);
}

template <typename T, typename U, int values_per_thread, int bits>
inline U grouped_load_vector(const device T* x, thread U* x_thread) {
  U sum = 0;
  if (bits == 4) {
    for (int i = 0; i < values_per_thread; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 16.0f;
      x_thread[i + 2] = x[i + 2] / 256.0f;
      x_thread[i + 3] = x[i + 3] / 4096.0f;
    }
  } else if (bits == 5) {
    for (int i = 0; i < values_per_thread; i += 8) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
          x[i + 6] + x[i + 7];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 32.0f;
      x_thread[i + 2] = x[i + 2] / 4.0f;
      x_thread[i + 3] = x[i + 3] / 128.0f;
      x_thread[i + 4] = x[i + 4] / 16.0f;
      x_thread[i + 5] = x[i + 5] / 2.0f;
      x_thread[i + 6] = x[i + 6] / 64.0f;
      x_thread[i + 7] = x[i + 7] / 8.0f;
    }
  } else if (bits == 6) {
    for (int i = 0; i < values_per_thread; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 64.0f;
      x_thread[i + 2] = x[i + 2] / 16.0f;
      x_thread[i + 3] = x[i + 3] / 4.0f;
    }
  } else if (bits == 8) {
    for (int i = 0; i < values_per_thread; i++) {
      sum += x[i];
      x_thread[i] = x[i];
    }
  }
  return sum;
}

template <int bits, int values_per_thread>
inline constexpr int grouped_term_count() {
  return bits == 5 ? (values_per_thread / 8) * 12
                   : (bits == 6 ? (values_per_thread / 4) * 6 : values_per_thread);
}

// The integer weight terms of MLX `qdot`, unpacked once per weight block.
// Every term is the exact masked integer MLX multiplies by an x value.
template <int bits, int values_per_thread>
inline void grouped_unpack(const device uint8_t* w, thread ushort* t) {
  if (bits == 4) {
    const device uint16_t* ws = (const device uint16_t*)w;
    for (int i = 0; i < (values_per_thread / 4); i++) {
      t[4 * i] = ws[i] & 0x000f;
      t[4 * i + 1] = ws[i] & 0x00f0;
      t[4 * i + 2] = ws[i] & 0x0f00;
      t[4 * i + 3] = ws[i] & 0xf000;
    }
  } else if (bits == 5) {
    for (int g = 0; g < (values_per_thread / 8); g++) {
      const device uint8_t* wg = w + 5 * g;
      thread ushort* tg = t + 12 * g;
      tg[0] = wg[0] & 0x1f;
      tg[1] = wg[0] & 0xe0;
      tg[2] = wg[1] & 0x3;
      tg[3] = wg[1] & 0x7c;
      tg[4] = wg[1] & 0x80;
      tg[5] = wg[2] & 0xf;
      tg[6] = wg[2] & 0xf0;
      tg[7] = wg[3] & 0x1;
      tg[8] = wg[3] & 0x3e;
      tg[9] = wg[3] & 0xc0;
      tg[10] = wg[4] & 0x7;
      tg[11] = wg[4] & 0xf8;
    }
  } else if (bits == 6) {
    for (int g = 0; g < (values_per_thread / 4); g++) {
      const device uint8_t* wg = w + 3 * g;
      thread ushort* tg = t + 6 * g;
      tg[0] = wg[0] & 0x3f;
      tg[1] = wg[0] & 0xc0;
      tg[2] = wg[1] & 0x0f;
      tg[3] = wg[1] & 0xf0;
      tg[4] = wg[2] & 0x03;
      tg[5] = wg[2] & 0xfc;
    }
  } else if (bits == 8) {
    for (int i = 0; i < values_per_thread; i++) {
      t[i] = w[i];
    }
  }
}

// MLX `qdot` evaluated over unpacked terms, statement for statement.
template <typename U, int values_per_thread, int bits>
inline U grouped_qdot_unpacked(
    const thread ushort* t,
    const thread U* x_thread,
    U scale,
    U bias,
    U sum) {
  U accum = 0;
  if (bits == 4) {
    for (int i = 0; i < (values_per_thread / 4); i++) {
      accum +=
          (x_thread[4 * i] * t[4 * i] + x_thread[4 * i + 1] * t[4 * i + 1] +
           x_thread[4 * i + 2] * t[4 * i + 2] + x_thread[4 * i + 3] * t[4 * i + 3]);
    }
  } else if (bits == 5) {
    for (int g = 0; g < (values_per_thread / 8); g++) {
      const thread U* xg = x_thread + 8 * g;
      const thread ushort* tg = t + 12 * g;
      accum += tg[0] * xg[0];
      accum += tg[1] * xg[1];
      accum += tg[2] * (xg[1] * 256.0f);
      accum += tg[3] * xg[2];
      accum += tg[4] * xg[3];
      accum += tg[5] * (xg[3] * 256.0f);
      accum += tg[6] * xg[4];
      accum += tg[7] * (xg[4] * 256.0f);
      accum += tg[8] * xg[5];
      accum += tg[9] * xg[6];
      accum += tg[10] * (xg[6] * 256.0f);
      accum += tg[11] * xg[7];
    }
  } else if (bits == 6) {
    for (int g = 0; g < (values_per_thread / 4); g++) {
      const thread U* xg = x_thread + 4 * g;
      const thread ushort* tg = t + 6 * g;
      accum += tg[0] * xg[0];
      accum += tg[1] * xg[1];
      accum += tg[2] * (xg[1] * 256.0f);
      accum += tg[3] * xg[2];
      accum += tg[4] * (xg[2] * 256.0f);
      accum += tg[5] * xg[3];
    }
  } else if (bits == 8) {
    for (int i = 0; i < values_per_thread; i++) {
      accum += x_thread[i] * t[i];
    }
  }
  return scale * accum + sum * bias;
}
