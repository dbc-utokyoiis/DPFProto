#pragma once

#include <iostream>
#include <algorithm> // std::max, std::min
#include <limits>    // std::numeric_limits
#include <vector>
#include <iomanip>   // std::boolalpha

template <typename T>
struct ColumnStats {
    T min_val = std::numeric_limits<T>::max();
    T max_val = std::numeric_limits<T>::lowest();
    bool initialized = false;

    void update(const T& val) {
        if (!initialized) {
            min_val = val;
            max_val = val;
            initialized = true;
        } else {
            if (val < min_val) min_val = val;
            if (val > max_val) max_val = val;
        }
    }
    
    void reset() {
        min_val = std::numeric_limits<T>::max();
        max_val = std::numeric_limits<T>::lowest();
        initialized = false;
    }

    [[nodiscard]] constexpr bool is_valid() const {
        return min_val <= max_val;
    }

    [[nodiscard]] constexpr bool overlaps(const T& query_low, const T& query_high) const {
        if (!is_valid()) return false;
        return std::max(min_val, query_low) <= std::min(max_val, query_high);
    }
};

/* Simplified type for storage */
template <typename T>
struct Stats {
    T min_val = std::numeric_limits<T>::max();
    T max_val = std::numeric_limits<T>::lowest();

    static Stats from_column_stats(const ColumnStats<T>& val) {
        return Stats{ .min_val = val.min_val, .max_val = val.max_val };
    }

    [[nodiscard]] constexpr bool is_valid() const {
        return min_val <= max_val;
    }

    [[nodiscard]] constexpr bool overlaps(const T& q) const {
        if (!is_valid()) return false;
        return std::max(min_val, q) <= std::min(max_val, q);
    }

    [[nodiscard]] constexpr bool overlaps(const T& query_low, const T& query_high) const {
        if (!is_valid()) return false;
        return std::max(min_val, query_low) <= std::min(max_val, query_high);
    }
};