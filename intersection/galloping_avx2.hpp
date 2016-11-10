#ifndef GALLOPING_AVX2_HPP_
#define GALLOPING_AVX2_HPP_


#ifdef __GNUC__
#define COMPILER_LIKELY(x) __builtin_expect((x), 1)
#define COMPILER_RARELY(x) __builtin_expect((x), 0)
#else
#define COMPILER_LIKELY(x) x
#define COMPILER_RARELY(x) x
#endif

#ifdef __AVX2__

size_t v1_avx2
(const uint32_t *rare, size_t lenRare,
 const uint32_t *freq, size_t lenFreq,
 uint32_t *matchOut) {
    assert(lenRare <= lenFreq);
    const uint32_t *matchOrig = matchOut;
    if (lenFreq == 0 || lenRare == 0) return 0;

    const uint64_t kFreqSpace = 2 * 4 * (0 + 1) - 1;
    const uint64_t kRareSpace = 0;

    const uint32_t *stopFreq = &freq[lenFreq] - kFreqSpace;
    const uint32_t *stopRare = &rare[lenRare] - kRareSpace;

    __m256i  Rare;

    __m256i F;

    if (COMPILER_RARELY( (rare >= stopRare) || (freq >= stopFreq) )) goto FINISH_SCALAR;
    uint32_t valRare;
    valRare = rare[0];
    Rare = _mm256_set1_epi32(valRare);

    uint64_t maxFreq;
    maxFreq = freq[2 * 4 - 1];
    F = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(freq));


    if (COMPILER_RARELY(maxFreq < valRare)) goto ADVANCE_FREQ;

ADVANCE_RARE:
    do {
        *matchOut = valRare;
        valRare = rare[1]; // for next iteration
        rare += 1;
        if (COMPILER_RARELY(rare >= stopRare)) {
            rare -= 1;
            goto FINISH_SCALAR;
        }
        F =  _mm256_cmpeq_epi32(F,Rare);
        Rare = _mm256_set1_epi32(valRare);
        if(_mm256_testz_si256(F,F) == 0)
          matchOut ++;
        F = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(freq));

    } while (maxFreq >= valRare);

    uint64_t maxProbe;

ADVANCE_FREQ:
    do {
        const uint64_t kProbe = (0 + 1) * 2 * 4;
        const uint32_t *probeFreq = freq + kProbe;
        maxProbe = freq[(0 + 2) * 2 * 4 - 1];

        if (COMPILER_RARELY(probeFreq >= stopFreq)) {
            goto FINISH_SCALAR;
        }

        freq = probeFreq;

    } while (maxProbe < valRare);

    maxFreq = maxProbe;

    F = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(freq));


    goto ADVANCE_RARE;

    size_t count;
FINISH_SCALAR:
    count = matchOut - matchOrig;

    lenFreq = stopFreq + kFreqSpace - freq;
    lenRare = stopRare + kRareSpace - rare;

    size_t tail = match_scalar(freq, lenFreq, rare, lenRare, matchOut);

    return count + tail;
}

size_t v3_avx2(const uint32_t *rare, const size_t lenRare,
        const uint32_t *freq, const size_t lenFreq, uint32_t * out) {
    if (lenFreq == 0 || lenRare == 0)
        return 0;
    assert(lenRare <= lenFreq);
    const uint32_t * const initout (out);
    typedef __m256i vec;
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
        const uint32_t matchRare = *rare;//nextRare;
        const vec Match = _mm256_set1_epi32(matchRare);
        while (freq[veclen * 31 + vecmax] < matchRare) { // if no match possible
            freq += veclen * 32; // advance 32 vectors
            if (freq > stopFreq)
                goto FINISH_SCALAR;
        }
        vec Q0,Q1,Q2,Q3;
        if(freq[veclen * 15 + vecmax] >= matchRare  ) {
        if(freq[veclen * 7 + vecmax] < matchRare  ) {
            Q0 = _mm256_or_si256(
            		_mm256_cmpeq_epi32(_mm256_loadu_si256((vec *) freq + 8), Match),
					_mm256_cmpeq_epi32(_mm256_loadu_si256((vec *) freq + 9), Match));
            Q1 = _mm256_or_si256(
            		_mm256_cmpeq_epi32(_mm256_loadu_si256((vec *) freq + 10), Match),
					_mm256_cmpeq_epi32(_mm256_loadu_si256((vec *) freq + 11), Match));

            Q2 = _mm256_or_si256(
            		_mm256_cmpeq_epi32(_mm256_loadu_si256((vec *) freq + 12), Match),
					_mm256_cmpeq_epi32(_mm256_loadu_si256((vec *) freq + 13), Match));
            Q3 = _mm256_or_si256(
            		_mm256_cmpeq_epi32(_mm256_loadu_si256((vec *) freq + 14), Match),
					_mm256_cmpeq_epi32(_mm256_loadu_si256((vec *) freq + 15), Match));
        } else {
            Q0 = _mm256_or_si256(
            		_mm256_cmpeq_epi32(_mm256_loadu_si256((vec *) freq + 4), Match),
					_mm256_cmpeq_epi32(_mm256_loadu_si256((vec *) freq + 5), Match));
            Q1 = _mm256_or_si256(
            		_mm256_cmpeq_epi32(_mm256_loadu_si256((vec *) freq + 6), Match),
					_mm256_cmpeq_epi32(_mm256_loadu_si256((vec *) freq + 7), Match));
            Q2 = _mm256_or_si256(
            		_mm256_cmpeq_epi32(_mm256_loadu_si256((vec *) freq + 0), Match),
					_mm256_cmpeq_epi32(_mm256_loadu_si256((vec *) freq + 1), Match));
            Q3 = _mm256_or_si256(
            		_mm256_cmpeq_epi32(_mm256_loadu_si256((vec *) freq + 2), Match),
					_mm256_cmpeq_epi32(_mm256_loadu_si256((vec *) freq + 3), Match));
        }
        }
        else
        {
            if(freq[veclen * 23 + vecmax] < matchRare  ) {
                Q0 = _mm256_or_si256(
                		_mm256_cmpeq_epi32(_mm256_loadu_si256((vec *) freq + 8 + 16), Match),
						_mm256_cmpeq_epi32(_mm256_loadu_si256((vec *) freq + 9 + 16), Match));
                Q1 = _mm256_or_si256(
                		_mm256_cmpeq_epi32(_mm256_loadu_si256((vec *) freq + 10+ 16), Match),
						_mm256_cmpeq_epi32(_mm256_loadu_si256((vec *) freq + 11+ 16), Match));

                Q2 = _mm256_or_si256(
                		_mm256_cmpeq_epi32(_mm256_loadu_si256((vec *) freq + 12+ 16), Match),
						_mm256_cmpeq_epi32(_mm256_loadu_si256((vec *) freq + 13+ 16), Match));
                Q3 = _mm256_or_si256(
                		_mm256_cmpeq_epi32(_mm256_loadu_si256((vec *) freq + 14+ 16), Match),
						_mm256_cmpeq_epi32(_mm256_loadu_si256((vec *) freq + 15+ 16), Match));
            } else {
                Q0 = _mm256_or_si256(
                		_mm256_cmpeq_epi32(_mm256_loadu_si256((vec *) freq + 4+ 16), Match),
						_mm256_cmpeq_epi32(_mm256_loadu_si256((vec *) freq + 5+ 16), Match));
                Q1 = _mm256_or_si256(
                		_mm256_cmpeq_epi32(_mm256_loadu_si256((vec *) freq + 6+ 16), Match),
						_mm256_cmpeq_epi32(_mm256_loadu_si256((vec *) freq + 7+ 16), Match));
                Q2 = _mm256_or_si256(
                		_mm256_cmpeq_epi32(_mm256_loadu_si256((vec *) freq + 0+ 16), Match),
						_mm256_cmpeq_epi32(_mm256_loadu_si256((vec *) freq + 1+ 16), Match));
                Q3 = _mm256_or_si256(
                		_mm256_cmpeq_epi32(_mm256_loadu_si256((vec *) freq + 2+ 16), Match),
						_mm256_cmpeq_epi32(_mm256_loadu_si256((vec *) freq + 3+ 16), Match));
            }

        }
        const vec F0 = _mm256_or_si256(_mm256_or_si256(Q0, Q1),_mm256_or_si256(Q2, Q3));
        if (_mm256_testz_si256(F0, F0)) {
        } else {
            *out++ = matchRare;
        }
    }

    FINISH_SCALAR: return (out - initout) + scalar(freq,
            stopFreq + freqspace - freq, rare, stopRare + rarespace - rare, out);
}

size_t SIMDgalloping_avx2(const uint32_t *rare, const size_t lenRare,
        const uint32_t *freq, const size_t lenFreq, uint32_t * out) {
    if (lenFreq == 0 || lenRare == 0)
        return 0;
    assert(lenRare <= lenFreq);
    const uint32_t * const initout (out);
    typedef __m256i vec;
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
        const uint32_t matchRare = *rare;//nextRare;
        const vec Match = _mm256_set1_epi32(matchRare);

        if (freq[veclen * 31 + vecmax] < matchRare) { // if no match possible
            uint32_t offset = 1;
            if (freq + veclen  * 32 > stopFreq) {
                freq += veclen * 32;
                goto FINISH_SCALAR;
            }
            while (freq[veclen * offset * 32 + veclen * 31 + vecmax]
                    < matchRare) { // if no match possible
                if (freq + veclen * (2 * offset ) * 32 <= stopFreq) {
                    offset *= 2;
                } else if (freq + veclen * (offset + 1) * 32 <= stopFreq) {
                    offset = static_cast<uint32_t>((stopFreq - freq ) / (veclen * 32));
                    //offset += 1;
                    if (freq[veclen * offset * 32 + veclen * 31 + vecmax]
                                    < matchRare) {
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
                if (freq[veclen * mid * 32 + veclen * 31 + vecmax]
                        < matchRare)
                    lower = mid;
                else
                    offset = mid;
            }
            freq += veclen * offset * 32;
        }
        vec Q0,Q1,Q2,Q3;
        if (freq[veclen * 15 + vecmax] >= matchRare) {
            if (freq[veclen * 7 + vecmax] < matchRare) {
                Q0
                        = _mm256_or_si256(
                                _mm256_cmpeq_epi32(
                                        _mm256_loadu_si256((vec *) freq + 8), Match),
                                _mm256_cmpeq_epi32(
                                        _mm256_loadu_si256((vec *) freq + 9), Match));
                Q1 = _mm256_or_si256(
                        _mm256_cmpeq_epi32(_mm256_loadu_si256((vec *) freq + 10),
                                Match),
                        _mm256_cmpeq_epi32(_mm256_loadu_si256((vec *) freq + 11),
                                Match));

                Q2 = _mm256_or_si256(
                        _mm256_cmpeq_epi32(_mm256_loadu_si256((vec *) freq + 12),
                                Match),
                        _mm256_cmpeq_epi32(_mm256_loadu_si256((vec *) freq + 13),
                                Match));
                Q3 = _mm256_or_si256(
                        _mm256_cmpeq_epi32(_mm256_loadu_si256((vec *) freq + 14),
                                Match),
                        _mm256_cmpeq_epi32(_mm256_loadu_si256((vec *) freq + 15),
                                Match));
            } else {
                Q0
                        = _mm256_or_si256(
                                _mm256_cmpeq_epi32(
                                        _mm256_loadu_si256((vec *) freq + 4), Match),
                                _mm256_cmpeq_epi32(
                                        _mm256_loadu_si256((vec *) freq + 5), Match));
                Q1
                        = _mm256_or_si256(
                                _mm256_cmpeq_epi32(
                                        _mm256_loadu_si256((vec *) freq + 6), Match),
                                _mm256_cmpeq_epi32(
                                        _mm256_loadu_si256((vec *) freq + 7), Match));
                Q2
                        = _mm256_or_si256(
                                _mm256_cmpeq_epi32(
                                        _mm256_loadu_si256((vec *) freq + 0), Match),
                                _mm256_cmpeq_epi32(
                                        _mm256_loadu_si256((vec *) freq + 1), Match));
                Q3
                        = _mm256_or_si256(
                                _mm256_cmpeq_epi32(
                                        _mm256_loadu_si256((vec *) freq + 2), Match),
                                _mm256_cmpeq_epi32(
                                        _mm256_loadu_si256((vec *) freq + 3), Match));
            }
        } else {
            if (freq[veclen * 23 + vecmax] < matchRare) {
                Q0 = _mm256_or_si256(
                        _mm256_cmpeq_epi32(_mm256_loadu_si256((vec *) freq + 8 + 16),
                                Match),
                        _mm256_cmpeq_epi32(_mm256_loadu_si256((vec *) freq + 9 + 16),
                                Match));
                Q1 = _mm256_or_si256(
                        _mm256_cmpeq_epi32(_mm256_loadu_si256((vec *) freq + 10 + 16),
                                Match),
                        _mm256_cmpeq_epi32(_mm256_loadu_si256((vec *) freq + 11 + 16),
                                Match));

                Q2 = _mm256_or_si256(
                        _mm256_cmpeq_epi32(_mm256_loadu_si256((vec *) freq + 12 + 16),
                                Match),
                        _mm256_cmpeq_epi32(_mm256_loadu_si256((vec *) freq + 13 + 16),
                                Match));
                Q3 = _mm256_or_si256(
                        _mm256_cmpeq_epi32(_mm256_loadu_si256((vec *) freq + 14 + 16),
                                Match),
                        _mm256_cmpeq_epi32(_mm256_loadu_si256((vec *) freq + 15 + 16),
                                Match));
            } else {
                Q0 = _mm256_or_si256(
                        _mm256_cmpeq_epi32(_mm256_loadu_si256((vec *) freq + 4 + 16),
                                Match),
                        _mm256_cmpeq_epi32(_mm256_loadu_si256((vec *) freq + 5 + 16),
                                Match));
                Q1 = _mm256_or_si256(
                        _mm256_cmpeq_epi32(_mm256_loadu_si256((vec *) freq + 6 + 16),
                                Match),
                        _mm256_cmpeq_epi32(_mm256_loadu_si256((vec *) freq + 7 + 16),
                                Match));
                Q2 = _mm256_or_si256(
                        _mm256_cmpeq_epi32(_mm256_loadu_si256((vec *) freq + 0 + 16),
                                Match),
                        _mm256_cmpeq_epi32(_mm256_loadu_si256((vec *) freq + 1 + 16),
                                Match));
                Q3 = _mm256_or_si256(
                        _mm256_cmpeq_epi32(_mm256_loadu_si256((vec *) freq + 2 + 16),
                                Match),
                        _mm256_cmpeq_epi32(_mm256_loadu_si256((vec *) freq + 3 + 16),
                                Match));
            }

        }
        const vec F0 = _mm256_or_si256(_mm256_or_si256(Q0, Q1),_mm256_or_si256(Q2, Q3));
        if (_mm256_testz_si256(F0, F0)) {
        } else {
            *out++ = matchRare;
        }
    }

    FINISH_SCALAR: return (out - initout) + scalar(freq,
            stopFreq + freqspace - freq, rare, stopRare + rarespace - rare, out);
}


/**
 * Our main heuristic.
 *
 * The out pointer can be set1 if length1<=length2,
 * or else it can be set2 if length1>length2.
 */
size_t SIMDintersection_avx2(const uint32_t *set1, const size_t length1,
                        const uint32_t *set2, const size_t length2,
                        uint32_t *out) {
  if ((length1 == 0) or (length2 == 0))
    return 0;

  if ((1000 * length1 <= length2) or (1000 * length2 <= length1)) {
    if (length1 <= length2)
      return SIMDgalloping_avx2(set1, length1, set2, length2, out);
    else
      return SIMDgalloping_avx2(set2, length2, set1, length1, out);
  }

  if ((50 * length1 <= length2) or (50 * length2 <= length1)) {
    if (length1 <= length2)
      return v3_avx2(set1, length1, set2, length2, out);
    else
      return v3_avx2(set2, length2, set1, length1, out);
  }

  if (length1 <= length2)
    return v1_avx2(set1, length1, set2, length2, out);
  else
    return v1_avx2(set2, length2, set1, length1, out);
}
#endif

#endif
