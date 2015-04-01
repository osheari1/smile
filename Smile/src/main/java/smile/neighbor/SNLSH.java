/**
 * ****************************************************************************
 * Copyright (c) 2010 Haifeng Li
 * <p/>
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * <p/>
 * http://www.apache.org/licenses/LICENSE-2.0
 * <p/>
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * *****************************************************************************
 */
package smile.neighbor;

import com.google.common.base.Charsets;
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import com.google.common.hash.HashFunction;
import com.google.common.hash.Hashing;
import smile.math.distance.HammingDistance;
import smile.util.MaxHeap;

import java.lang.reflect.Array;
import java.util.Arrays;
import java.util.Set;
import java.util.List;
import java.util.LinkedList;
import java.util.HashSet;
import java.util.LinkedHashMap;



/**
 *
 * Locality-Sensitive Hashing for Signatures.
 * LSH is an efficient algorithm for approximate nearest neighbor search
 * in high dimensional spaces by performing probabilistic dimension reduction of data.
 * The basic idea is to hash the input items so that similar items are mapped to the same
 * buckets with high probability (the number of buckets being much smaller
 * than the universe of possible input items).
 * To avoid computing the similarity of every pair of sets or their signatures.
 * If we are given signatures for the sets, we may divide them into bands, and only
 * measure the similarity of a pair of sets if they are identical in at least one band.
 * By choosing the size of bands appropriately, we can eliminate from
 * consideration most of the pairs that do not meet our threshold of similarity.
 *
 * <h2>References</h2>
 * <ol>
 * <li>Moses S. Charikar. Similarity Estimation Techniques from Rounding Algorithms</li>
 * <li> Alexis Joly and Olivier Buisson. A posteriori multi-probe locality sensitive hashing. ACM international conference on Multimedia, 2008. </li>
 * </ol>
 *
 * @see LSH
 * @author Qiyang Zuo
 *
 */
public class SNLSH<E> implements NearestNeighborSearch<String, E>, KNNSearch<String, E>, RNNSearch<String, E> {


    private final int bandSize;
    private final long mask;
    private static final int BITS = 64;
    /**
     * Signature fractions
     */
    private Band[] bands;
    /**
     * universal hash function
     */
    private static HashFunction hf = Hashing.murmur3_128();
    /**
     * The data objects.
     */
    private List<E> data;
    /**
     * The keys of data objects.
     */
    private List<String> keys;
    /**
     * signatures generated by simhash
     */
    private List<Long> signs;

    private final int shingleSize;
    /**
     * Whether to exclude query object self from the neighborhood.
     */
    private boolean identicalExcluded = true;

    @SuppressWarnings("unchecked")
    public SNLSH(int bandSize, int shingleSize) {
        if (bandSize < 2 || bandSize > 32) {
            throw new IllegalArgumentException("Invalid band size!");
        }
        this.bandSize = bandSize;
        bands = (Band[]) Array.newInstance(Band.class, bandSize);
        Arrays.fill(bands, new Band());
        this.mask = -1 >>> (BITS / bandSize * (bandSize - 1));
        data = Lists.newArrayList();
        keys = Lists.newArrayList();
        signs = Lists.newArrayList();
        this.shingleSize = shingleSize;
    }

    public void put(String k, E v) {
        int index = data.size();
        data.add(v);
        keys.add(k);
        long sign = simhash64(k);
        signs.add(sign);
        for (int i = 0; i < bands.length; i++) {
            long bandKey = bandHash(sign, i);
            Bucket bucket = bands[i].get(bandKey);
            if (bucket == null) {
                bucket = new Bucket();
            }
            bucket.add(index);
            bands[i].put(bandKey, bucket);
        }
    }

    public Neighbor<String, E>[] knn(String q, int k) {
        if(k < 1) {
            throw new IllegalArgumentException("Invalid k: " + k);
        }
        long fpq = simhash64(q);
        Set<Integer> candidates = obtainCandidates(q);
        @SuppressWarnings("unchecked")
        Neighbor<String, E>[] neighbors = (Neighbor<String, E>[])Array.newInstance(Neighbor.class, k);
        MaxHeap<Neighbor<String, E>> heap = new MaxHeap<Neighbor<String, E>>(neighbors);
        for (int index : candidates) {
            long sign = signs.get(index);
            double distance = HammingDistance.d(fpq, sign);
            if (!keys.get(index).equals(q) && identicalExcluded) {
                heap.add(new Neighbor<String, E>(keys.get(index), data.get(index), index, distance));
            }
        }
        return heap.toSortedArray();
    }

    public Neighbor<String, E> nearest(String q) {
        Neighbor<String, E>[] ns = knn(q, 1);
        if(ns.length>0) {
            return ns[0];
        }
        return new Neighbor<String, E>(null, null, -1, Double.MAX_VALUE);
    }

    public void range(String q, double radius, List<Neighbor<String, E>> neighbors) {
        if (radius <= 0.0) {
            throw new IllegalArgumentException("Invalid radius: " + radius);
        }
        long fpq = simhash64(q);
        Set<Integer> candidates = obtainCandidates(q);
        for (int index : candidates) {
            double distance = HammingDistance.d(fpq, signs.get(index));
            if (distance <= radius) {
                if (keys.get(index).equals(q) && identicalExcluded) {
                    continue;
                }
                neighbors.add(new Neighbor<String, E>(keys.get(index), data.get(index), index, distance));
            }
        }
    }

    private class Band extends LinkedHashMap<Long, Bucket> {}

    private class Bucket extends LinkedList<Integer> {}

    private long bandHash(long hash, int bandNum) {
        return hash >>> ((bandNum * (BITS / this.bandSize))) & mask;
    }


    private long simhash64(String text) {
        Set<String> shingles = shingling(text, shingleSize);
        return simhash64(shingles);
    }

    private long simhash64(Set<String> shingles) {
        if (shingles == null || shingles.isEmpty()) {
            return 0;
        }
        int[] bits = new int[BITS];
        for (String s : shingles) {
            long hc = hf.hashString(s, Charsets.UTF_8).padToLong();
            for (int i = 0; i < BITS; i++) {
                if (((hc >>> i) & 1) == 1) {
                    bits[i]++;
                } else {
                    bits[i]--;
                }
            }
        }
        long hash = 0;
        long one = 1;
        for (int i = 0; i < BITS; i++) {
            if (bits[i] >= 0) {
                hash |= one;
            }
            one <<= 1;
        }
        return hash;
    }

    private static Set<String> shingling(String text, int shingleSize) {
        if (shingleSize < 1) {
            throw new IllegalArgumentException("Invalid shingle size!");
        }
        Set<String> shingles = Sets.newHashSet();
        if (text.length() <= shingleSize) {
            shingles.add(text);
        } else {
            for (int i = 0; i + shingleSize <= text.length(); i++) {
                shingles.add(text.substring(i, i + shingleSize));
            }
        }
//        shingles.addAll(Splitter.on(" ").omitEmptyStrings().splitToList(text));
        return shingles;
    }

    private Set<Integer> obtainCandidates(String q) {
        Set<Integer> candidates = new HashSet<Integer>();
        long sign = simhash64(q);
        for (int i = 0; i < bands.length; i++) {
            long bandKey = bandHash(sign, i);
            Bucket bucket = bands[i].get(bandKey);
            if (bucket != null) {
                candidates.addAll(bucket);
            }
        }
        return candidates;
    }
}
