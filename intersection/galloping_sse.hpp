#ifndef GALLOPING_SSE_HPP_
#define GALLOPING_SSE_HPP_


// copied from:
// https://github.com/lemire/SIMDCompressionAndIntersection

//TODO: use our scalar version
/**
 * Fast scalar scheme designed by N. Kurz.
 */
size_t scalar(const uint32_t *A, const size_t lenA, const uint32_t *B,
              const size_t lenB, uint32_t *out) {
  const uint32_t *const initout(out);
  if (lenA == 0 || lenB == 0)
    return 0;

  const uint32_t *endA = A + lenA;
  const uint32_t *endB = B + lenB;

  while (1) {
    while (*A < *B) {
    SKIP_FIRST_COMPARE:
      if (++A == endA)
        return (out - initout);
    }
    while (*A > *B) {
      if (++B == endB)
        return (out - initout);
    }
    if (*A == *B) {
      *out++ = *A;
      if (++A == endA || ++B == endB)
        return (out - initout);
    } else {
      goto SKIP_FIRST_COMPARE;
    }
  }

  return (out - initout); // NOTREACHED
}


size_t match_scalar(const uint32_t *A, const size_t lenA, const uint32_t *B,
                    const size_t lenB, uint32_t *out) {

  const uint32_t *initout = out;
  if (lenA == 0 || lenB == 0)
    return 0;

  const uint32_t *endA = A + lenA;
  const uint32_t *endB = B + lenB;

  while (1) {
    while (*A < *B) {
    SKIP_FIRST_COMPARE:
      if (++A == endA)
        goto FINISH;
    }
    while (*A > *B) {
      if (++B == endB)
        goto FINISH;
    }
    if (*A == *B) {
      *out++ = *A;
      if (++A == endA || ++B == endB)
        goto FINISH;
    } else {
      goto SKIP_FIRST_COMPARE;
    }
  }

FINISH:
  return (out - initout);
}

#ifdef __GNUC__
#define COMPILER_LIKELY(x) __builtin_expect((x), 1)
#define COMPILER_RARELY(x) __builtin_expect((x), 0)
#else
#define COMPILER_LIKELY(x) x
#define COMPILER_RARELY(x) x
#endif
/**
 * Intersections scheme designed by N. Kurz that works very
 * well when intersecting an array with another where the density
 * differential is small (between 2 to 10).
 *
 * It assumes that lenRare <= lenFreq.
 *
 * Note that this is not symmetric: flipping the rare and freq pointers
 * as well as lenRare and lenFreq could lead to significant performance
 * differences.
 *
 * The matchOut pointer can safely be equal to the rare pointer.
 *
 */
size_t v1(const uint32_t *rare, size_t lenRare, const uint32_t *freq,
          size_t lenFreq, uint32_t *matchOut) {
  assert(lenRare <= lenFreq);
  const uint32_t *matchOrig = matchOut;
  if (lenFreq == 0 || lenRare == 0)
    return 0;

  const uint64_t kFreqSpace = 2 * 4 * (0 + 1) - 1;
  const uint64_t kRareSpace = 0;

  const uint32_t *stopFreq = &freq[lenFreq] - kFreqSpace;
  const uint32_t *stopRare = &rare[lenRare] - kRareSpace;

  __m128i Rare;

  __m128i F0, F1;

  if (COMPILER_RARELY((rare >= stopRare) || (freq >= stopFreq)))
    goto FINISH_SCALAR;
  uint32_t valRare;
  valRare = rare[0];
  Rare = _mm_set1_epi32(valRare);

  uint64_t maxFreq;
  maxFreq = freq[2 * 4 - 1];
  F0 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(freq));
  F1 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(freq + 4));

  if (COMPILER_RARELY(maxFreq < valRare))
    goto ADVANCE_FREQ;

ADVANCE_RARE:
  do {
    *matchOut = valRare;
    rare += 1;
    if (COMPILER_RARELY(rare >= stopRare)) {
      rare -= 1;
      goto FINISH_SCALAR;
    }
    valRare = rare[0]; // for next iteration
    F0 = _mm_cmpeq_epi32(F0, Rare);
    F1 = _mm_cmpeq_epi32(F1, Rare);
    Rare = _mm_set1_epi32(valRare);
    F0 = _mm_or_si128(F0, F1);
#ifdef __SSE4_1__
    if (_mm_testz_si128(F0, F0) == 0)
      matchOut++;
#else
    if (_mm_movemask_epi8(F0))
      matchOut++;
#endif
    F0 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(freq));
    F1 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(freq + 4));

  } while (maxFreq >= valRare);

  uint64_t maxProbe;

ADVANCE_FREQ:
  do {
    const uint64_t kProbe = (0 + 1) * 2 * 4;
    const uint32_t *probeFreq = freq + kProbe;

    if (COMPILER_RARELY(probeFreq >= stopFreq)) {
      goto FINISH_SCALAR;
    }
    maxProbe = freq[(0 + 2) * 2 * 4 - 1];

    freq = probeFreq;

  } while (maxProbe < valRare);

  maxFreq = maxProbe;

  F0 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(freq));
  F1 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(freq + 4));

  goto ADVANCE_RARE;

  size_t count;
FINISH_SCALAR:
  count = matchOut - matchOrig;

  lenFreq = stopFreq + kFreqSpace - freq;
  lenRare = stopRare + kRareSpace - rare;

  size_t tail = match_scalar(freq, lenFreq, rare, lenRare, matchOut);

  return count + tail;
}

/**
 * This intersection function is similar to v1, but is faster when
 * the difference between lenRare and lenFreq is large, but not too large.

 * It assumes that lenRare <= lenFreq.
 *
 * Note that this is not symmetric: flipping the rare and freq pointers
 * as well as lenRare and lenFreq could lead to significant performance
 * differences.
 *
 * The matchOut pointer can safely be equal to the rare pointer.
 *
 * This function DOES NOT use inline assembly instructions. Just intrinsics.
 */
size_t v3(const uint32_t *rare, const size_t lenRare, const uint32_t *freq,
          const size_t lenFreq, uint32_t *out) {
  if (lenFreq == 0 || lenRare == 0)
    return 0;
  assert(lenRare <= lenFreq);
  const uint32_t *const initout(out);
  typedef __m128i vec;
  const uint32_t veclen = sizeof(vec) / sizeof(uint32_t);
  const size_t vecmax = veclen - 1;
  const size_t freqspace = 32 * veclen;
  const size_t rarespace = 1;

  const uint32_t *stopFreq = freq + lenFreq - freqspace;
  const uint32_t *stopRare = rare + lenRare - rarespace;
  if (freq > stopFreq) {
    return scalar(freq, lenFreq, rare, lenRare, out);
  }
  while (freq[veclen * 31 + vecmax] < *rare) {
    freq += veclen * 32;
    if (freq > stopFreq)
      goto FINISH_SCALAR;
  }
  for (; rare < stopRare; ++rare) {
    const uint32_t matchRare = *rare; // nextRare;
    const vec Match = _mm_set1_epi32(matchRare);
    while (freq[veclen * 31 + vecmax] < matchRare) { // if no match possible
      freq += veclen * 32;                           // advance 32 vectors
      if (freq > stopFreq)
        goto FINISH_SCALAR;
    }
    vec Q0, Q1, Q2, Q3;
    if (freq[veclen * 15 + vecmax] >= matchRare) {
      if (freq[veclen * 7 + vecmax] < matchRare) {
        Q0 = _mm_or_si128(
            _mm_cmpeq_epi32(
                _mm_loadu_si128(reinterpret_cast<const vec *>(freq) + 8),
                Match),
            _mm_cmpeq_epi32(
                _mm_loadu_si128(reinterpret_cast<const vec *>(freq) + 9),
                Match));
        Q1 = _mm_or_si128(
            _mm_cmpeq_epi32(
                _mm_loadu_si128(reinterpret_cast<const vec *>(freq) + 10),
                Match),
            _mm_cmpeq_epi32(
                _mm_loadu_si128(reinterpret_cast<const vec *>(freq) + 11),
                Match));

        Q2 = _mm_or_si128(
            _mm_cmpeq_epi32(
                _mm_loadu_si128(reinterpret_cast<const vec *>(freq) + 12),
                Match),
            _mm_cmpeq_epi32(
                _mm_loadu_si128(reinterpret_cast<const vec *>(freq) + 13),
                Match));
        Q3 = _mm_or_si128(
            _mm_cmpeq_epi32(
                _mm_loadu_si128(reinterpret_cast<const vec *>(freq) + 14),
                Match),
            _mm_cmpeq_epi32(
                _mm_loadu_si128(reinterpret_cast<const vec *>(freq) + 15),
                Match));
      } else {
        Q0 = _mm_or_si128(
            _mm_cmpeq_epi32(
                _mm_loadu_si128(reinterpret_cast<const vec *>(freq) + 4),
                Match),
            _mm_cmpeq_epi32(
                _mm_loadu_si128(reinterpret_cast<const vec *>(freq) + 5),
                Match));
        Q1 = _mm_or_si128(
            _mm_cmpeq_epi32(
                _mm_loadu_si128(reinterpret_cast<const vec *>(freq) + 6),
                Match),
            _mm_cmpeq_epi32(
                _mm_loadu_si128(reinterpret_cast<const vec *>(freq) + 7),
                Match));
        Q2 = _mm_or_si128(
            _mm_cmpeq_epi32(
                _mm_loadu_si128(reinterpret_cast<const vec *>(freq) + 0),
                Match),
            _mm_cmpeq_epi32(
                _mm_loadu_si128(reinterpret_cast<const vec *>(freq) + 1),
                Match));
        Q3 = _mm_or_si128(
            _mm_cmpeq_epi32(
                _mm_loadu_si128(reinterpret_cast<const vec *>(freq) + 2),
                Match),
            _mm_cmpeq_epi32(
                _mm_loadu_si128(reinterpret_cast<const vec *>(freq) + 3),
                Match));
      }
    } else {
      if (freq[veclen * 23 + vecmax] < matchRare) {
        Q0 = _mm_or_si128(
            _mm_cmpeq_epi32(
                _mm_loadu_si128(reinterpret_cast<const vec *>(freq) + 8 + 16),
                Match),
            _mm_cmpeq_epi32(
                _mm_loadu_si128(reinterpret_cast<const vec *>(freq) + 9 + 16),
                Match));
        Q1 = _mm_or_si128(
            _mm_cmpeq_epi32(
                _mm_loadu_si128(reinterpret_cast<const vec *>(freq) + 10 + 16),
                Match),
            _mm_cmpeq_epi32(
                _mm_loadu_si128(reinterpret_cast<const vec *>(freq) + 11 + 16),
                Match));

        Q2 = _mm_or_si128(
            _mm_cmpeq_epi32(
                _mm_loadu_si128(reinterpret_cast<const vec *>(freq) + 12 + 16),
                Match),
            _mm_cmpeq_epi32(
                _mm_loadu_si128(reinterpret_cast<const vec *>(freq) + 13 + 16),
                Match));
        Q3 = _mm_or_si128(
            _mm_cmpeq_epi32(
                _mm_loadu_si128(reinterpret_cast<const vec *>(freq) + 14 + 16),
                Match),
            _mm_cmpeq_epi32(
                _mm_loadu_si128(reinterpret_cast<const vec *>(freq) + 15 + 16),
                Match));
      } else {
        Q0 = _mm_or_si128(
            _mm_cmpeq_epi32(
                _mm_loadu_si128(reinterpret_cast<const vec *>(freq) + 4 + 16),
                Match),
            _mm_cmpeq_epi32(
                _mm_loadu_si128(reinterpret_cast<const vec *>(freq) + 5 + 16),
                Match));
        Q1 = _mm_or_si128(
            _mm_cmpeq_epi32(
                _mm_loadu_si128(reinterpret_cast<const vec *>(freq) + 6 + 16),
                Match),
            _mm_cmpeq_epi32(
                _mm_loadu_si128(reinterpret_cast<const vec *>(freq) + 7 + 16),
                Match));
        Q2 = _mm_or_si128(
            _mm_cmpeq_epi32(
                _mm_loadu_si128(reinterpret_cast<const vec *>(freq) + 0 + 16),
                Match),
            _mm_cmpeq_epi32(
                _mm_loadu_si128(reinterpret_cast<const vec *>(freq) + 1 + 16),
                Match));
        Q3 = _mm_or_si128(
            _mm_cmpeq_epi32(
                _mm_loadu_si128(reinterpret_cast<const vec *>(freq) + 2 + 16),
                Match),
            _mm_cmpeq_epi32(
                _mm_loadu_si128(reinterpret_cast<const vec *>(freq) + 3 + 16),
                Match));
      }
    }
    const vec F0 = _mm_or_si128(_mm_or_si128(Q0, Q1), _mm_or_si128(Q2, Q3));
#ifdef __SSE4_1__
    if (_mm_testz_si128(F0, F0)) {
#else
    if (!_mm_movemask_epi8(F0)) {
#endif
    } else {
      *out++ = matchRare;
    }
  }

FINISH_SCALAR:
  return (out - initout) + scalar(freq, stopFreq + freqspace - freq, rare,
                                  stopRare + rarespace - rare, out);
}

/**
 * This is the SIMD galloping function. This intersection function works well
 * when lenRare and lenFreq have vastly different values.
 *
 * It assumes that lenRare <= lenFreq.
 *
 * Note that this is not symmetric: flipping the rare and freq pointers
 * as well as lenRare and lenFreq could lead to significant performance
 * differences.
 *
 * The matchOut pointer can safely be equal to the rare pointer.
 *
 * This function DOES NOT use assembly. It only relies on intrinsics.
 */
size_t SIMDgalloping(const uint32_t *rare, const size_t lenRare,
                     const uint32_t *freq, const size_t lenFreq,
                     uint32_t *out) {
  if (lenFreq == 0 || lenRare == 0)
    return 0;
  assert(lenRare <= lenFreq);
  const uint32_t *const initout(out);
  typedef __m128i vec;
  const uint32_t veclen = sizeof(vec) / sizeof(uint32_t);
  const size_t vecmax = veclen - 1;
  const size_t freqspace = 32 * veclen;
  const size_t rarespace = 1;

  const uint32_t *stopFreq = freq + lenFreq - freqspace;
  const uint32_t *stopRare = rare + lenRare - rarespace;
  if (freq > stopFreq) {
    return scalar(freq, lenFreq, rare, lenRare, out);
  }
  for (; rare < stopRare; ++rare) {
    const uint32_t matchRare = *rare; // nextRare;
    const vec Match = _mm_set1_epi32(matchRare);

    if (freq[veclen * 31 + vecmax] < matchRare) { // if no match possible
      uint32_t offset = 1;
      if (freq + veclen * 32 > stopFreq) {
        freq += veclen * 32;
        goto FINISH_SCALAR;
      }
      while (freq[veclen * offset * 32 + veclen * 31 + vecmax] <
             matchRare) { // if no match possible
        if (freq + veclen * (2 * offset) * 32 <= stopFreq) {
          offset *= 2;
        } else if (freq + veclen * (offset + 1) * 32 <= stopFreq) {
          offset = static_cast<uint32_t>((stopFreq - freq) / (veclen * 32));
          // offset += 1;
          if (freq[veclen * offset * 32 + veclen * 31 + vecmax] < matchRare) {
            freq += veclen * offset * 32;
            goto FINISH_SCALAR;
          } else {
            break;
          }
        } else {
          freq += veclen * offset * 32;
          goto FINISH_SCALAR;
        }
      }
      uint32_t lower = offset / 2;
      while (lower + 1 != offset) {
        const uint32_t mid = (lower + offset) / 2;
        if (freq[veclen * mid * 32 + veclen * 31 + vecmax] < matchRare)
          lower = mid;
        else
          offset = mid;
      }
      freq += veclen * offset * 32;
    }
    vec Q0, Q1, Q2, Q3;
    if (freq[veclen * 15 + vecmax] >= matchRare) {
      if (freq[veclen * 7 + vecmax] < matchRare) {
        Q0 = _mm_or_si128(
            _mm_cmpeq_epi32(
                _mm_loadu_si128(reinterpret_cast<const vec *>(freq) + 8),
                Match),
            _mm_cmpeq_epi32(
                _mm_loadu_si128(reinterpret_cast<const vec *>(freq) + 9),
                Match));
        Q1 = _mm_or_si128(
            _mm_cmpeq_epi32(
                _mm_loadu_si128(reinterpret_cast<const vec *>(freq) + 10),
                Match),
            _mm_cmpeq_epi32(
                _mm_loadu_si128(reinterpret_cast<const vec *>(freq) + 11),
                Match));

        Q2 = _mm_or_si128(
            _mm_cmpeq_epi32(
                _mm_loadu_si128(reinterpret_cast<const vec *>(freq) + 12),
                Match),
            _mm_cmpeq_epi32(
                _mm_loadu_si128(reinterpret_cast<const vec *>(freq) + 13),
                Match));
        Q3 = _mm_or_si128(
            _mm_cmpeq_epi32(
                _mm_loadu_si128(reinterpret_cast<const vec *>(freq) + 14),
                Match),
            _mm_cmpeq_epi32(
                _mm_loadu_si128(reinterpret_cast<const vec *>(freq) + 15),
                Match));
      } else {
        Q0 = _mm_or_si128(
            _mm_cmpeq_epi32(
                _mm_loadu_si128(reinterpret_cast<const vec *>(freq) + 4),
                Match),
            _mm_cmpeq_epi32(
                _mm_loadu_si128(reinterpret_cast<const vec *>(freq) + 5),
                Match));
        Q1 = _mm_or_si128(
            _mm_cmpeq_epi32(
                _mm_loadu_si128(reinterpret_cast<const vec *>(freq) + 6),
                Match),
            _mm_cmpeq_epi32(
                _mm_loadu_si128(reinterpret_cast<const vec *>(freq) + 7),
                Match));
        Q2 = _mm_or_si128(
            _mm_cmpeq_epi32(
                _mm_loadu_si128(reinterpret_cast<const vec *>(freq) + 0),
                Match),
            _mm_cmpeq_epi32(
                _mm_loadu_si128(reinterpret_cast<const vec *>(freq) + 1),
                Match));
        Q3 = _mm_or_si128(
            _mm_cmpeq_epi32(
                _mm_loadu_si128(reinterpret_cast<const vec *>(freq) + 2),
                Match),
            _mm_cmpeq_epi32(
                _mm_loadu_si128(reinterpret_cast<const vec *>(freq) + 3),
                Match));
      }
    } else {
      if (freq[veclen * 23 + vecmax] < matchRare) {
        Q0 = _mm_or_si128(
            _mm_cmpeq_epi32(
                _mm_loadu_si128(reinterpret_cast<const vec *>(freq) + 8 + 16),
                Match),
            _mm_cmpeq_epi32(
                _mm_loadu_si128(reinterpret_cast<const vec *>(freq) + 9 + 16),
                Match));
        Q1 = _mm_or_si128(
            _mm_cmpeq_epi32(
                _mm_loadu_si128(reinterpret_cast<const vec *>(freq) + 10 + 16),
                Match),
            _mm_cmpeq_epi32(
                _mm_loadu_si128(reinterpret_cast<const vec *>(freq) + 11 + 16),
                Match));

        Q2 = _mm_or_si128(
            _mm_cmpeq_epi32(
                _mm_loadu_si128(reinterpret_cast<const vec *>(freq) + 12 + 16),
                Match),
            _mm_cmpeq_epi32(
                _mm_loadu_si128(reinterpret_cast<const vec *>(freq) + 13 + 16),
                Match));
        Q3 = _mm_or_si128(
            _mm_cmpeq_epi32(
                _mm_loadu_si128(reinterpret_cast<const vec *>(freq) + 14 + 16),
                Match),
            _mm_cmpeq_epi32(
                _mm_loadu_si128(reinterpret_cast<const vec *>(freq) + 15 + 16),
                Match));
      } else {
        Q0 = _mm_or_si128(
            _mm_cmpeq_epi32(
                _mm_loadu_si128(reinterpret_cast<const vec *>(freq) + 4 + 16),
                Match),
            _mm_cmpeq_epi32(
                _mm_loadu_si128(reinterpret_cast<const vec *>(freq) + 5 + 16),
                Match));
        Q1 = _mm_or_si128(
            _mm_cmpeq_epi32(
                _mm_loadu_si128(reinterpret_cast<const vec *>(freq) + 6 + 16),
                Match),
            _mm_cmpeq_epi32(
                _mm_loadu_si128(reinterpret_cast<const vec *>(freq) + 7 + 16),
                Match));
        Q2 = _mm_or_si128(
            _mm_cmpeq_epi32(
                _mm_loadu_si128(reinterpret_cast<const vec *>(freq) + 0 + 16),
                Match),
            _mm_cmpeq_epi32(
                _mm_loadu_si128(reinterpret_cast<const vec *>(freq) + 1 + 16),
                Match));
        Q3 = _mm_or_si128(
            _mm_cmpeq_epi32(
                _mm_loadu_si128(reinterpret_cast<const vec *>(freq) + 2 + 16),
                Match),
            _mm_cmpeq_epi32(
                _mm_loadu_si128(reinterpret_cast<const vec *>(freq) + 3 + 16),
                Match));
      }
    }
    const vec F0 = _mm_or_si128(_mm_or_si128(Q0, Q1), _mm_or_si128(Q2, Q3));
#ifdef __SSE4_1__
    if (_mm_testz_si128(F0, F0)) {
#else
    if (!_mm_movemask_epi8(F0)) {
#endif
    } else {
      *out++ = matchRare;
    }
  }

FINISH_SCALAR:
  return (out - initout) + scalar(freq, stopFreq + freqspace - freq, rare,
                                  stopRare + rarespace - rare, out);
}

/**
 * Our main heuristic.
 *
 * The out pointer can be set1 if length1<=length2,
 * or else it can be set2 if length1>length2.
 */
size_t SIMDintersection(const uint32_t *set1, const size_t length1,
                        const uint32_t *set2, const size_t length2,
                        uint32_t *out) {
  if ((length1 == 0) or (length2 == 0))
    return 0;

  if ((1000 * length1 <= length2) or (1000 * length2 <= length1)) {
    if (length1 <= length2)
      return SIMDgalloping(set1, length1, set2, length2, out);
    else
      return SIMDgalloping(set2, length2, set1, length1, out);
  }

  if ((50 * length1 <= length2) or (50 * length2 <= length1)) {
    if (length1 <= length2)
      return v3(set1, length1, set2, length2, out);
    else
      return v3(set2, length2, set1, length1, out);
  }

  if (length1 <= length2)
    return v1(set1, length1, set2, length2, out);
  else
    return v1(set2, length2, set1, length1, out);
}

#endif
