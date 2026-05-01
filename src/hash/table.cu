#include <cstdint>

#include <cuda/std/utility>

template <typename Key, typename Value>
class StaticHashTable {
private:
    /**
     * Although a ctrl is 8 bits in the Rust backend, we use 16 bits here,
     * because 8-bit CAS is not originally available in CUDA.
     */
    uint16_t *ctrl_;
    cuda::std::pair<Key, Value> *slots_;
    size_t capacity_;

public:
    static const uint16_t EMPTY = 0xffff;
    static const uint16_t UPDATING = 0xfffe;

    __device__ StaticHashTable(uint16_t *ctrl, cuda::std::pair<Key, Value> *slots, size_t capacity)
        : ctrl_(ctrl), slots_(slots), capacity_(capacity) {}

    template <typename Hash>
    __device__ void insert(Key &&key, Value &&value, Hash hasher)
    {
        uint64_t hash = hasher(key);
        uint16_t new_ctrl = hash & 0x7fff;
        size_t i = hash % this->capacity_;

        while (true) {
            uint16_t *ctrl = &this->ctrl_[i];

            uint16_t prev_ctrl = atomicCAS(ctrl, EMPTY, UPDATING);

            if (prev_ctrl == EMPTY) {
                this->slots_[i] = cuda::std::make_pair(key, value);
                // *ctrl = new_ctrl;
                asm("st.volatile.global.u16 [%0], %1;" ::"l"(ctrl), "h"(new_ctrl));
                return;
            }
            else {
                while (prev_ctrl == UPDATING) {
                    // prev_ctrl = *ctrl;
                    asm("ld.volatile.global.u16 %0, [%1];" : "=h"(prev_ctrl) : "l"(ctrl));
                }
                if (prev_ctrl == new_ctrl) {
                    cuda::std::pair<Key, Value> *slot = &this->slots_[i];
                    if (slot->first == key) {
                        slot->second = value;
                        return;
                    }
                }
            }

            i = (i + 1) % this->capacity_;
        }
    }

    template <typename Hash>
    __device__ Value *find(const Key &key, Hash hasher)
    {
        uint64_t hash = hasher(key);
        size_t i = hash % this->capacity_;

        while (true) {
            uint16_t found_ctrl = this->ctrl_[i];

            if (found_ctrl == EMPTY) {
                return nullptr;
            }
            else {
                uint16_t expected_ctrl = hash & 0x7fff;
                if (found_ctrl == expected_ctrl) {
                    cuda::std::pair<Key, Value> *slot = &this->slots_[i];
                    if (slot->first == key) {
                        return &slot->second;
                    }
                }
            }

            i = (i + 1) % this->capacity_;
        }
    }

    template <typename Hash>
    __device__ Value *find_or_insert(Key &&key, Value &&value, Hash hasher)
    {
        uint64_t hash = hasher(key);
        uint16_t new_ctrl = hash & 0x7fff;
        size_t i = hash % this->capacity_;

        while (true) {
            uint16_t *ctrl = &this->ctrl_[i];

            uint16_t prev_ctrl = atomicCAS(ctrl, EMPTY, UPDATING);

            if (prev_ctrl == EMPTY) {
                cuda::std::pair<Key, Value> *slot = &this->slots_[i];
                *slot = cuda::std::make_pair(key, value);
                // *ctrl = new_ctrl;
                asm("st.volatile.global.u16 [%0], %1;" ::"l"(ctrl), "h"(new_ctrl));
                return &slot->second;
            }
            else {
                while (prev_ctrl == UPDATING) {
                    // prev_ctrl = *ctrl;
                    asm("ld.volatile.global.u16 %0, [%1];" : "=h"(prev_ctrl) : "l"(ctrl));
                }
                if (prev_ctrl == new_ctrl) {
                    cuda::std::pair<Key, Value> *slot = &this->slots_[i];
                    if (slot->first == key) {
                        return &slot->second;
                    }
                }
            }

            i = (i + 1) % this->capacity_;
        }
    }
};
