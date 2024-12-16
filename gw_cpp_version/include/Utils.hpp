#pragma once

#include <vector>
#include <ctime>

template <typename T>
inline int findIndex(const std::vector<T>& vec, const T& value) {
    for (size_t i = 0; i < vec.size(); ++i) {
        if (vec[i] == value) {
            return static_cast<int>(i); // Return the index if the value is found
        }
    }
    return -1; // Return -1 if the value is not found
}

inline int getTimestampSec() {
    return static_cast<int>(std::time(nullptr));
}
