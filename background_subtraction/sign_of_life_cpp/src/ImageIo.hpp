/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Host-only I/O: JPEG decode (stb_image) into RGB, mask encode (stb_image_write)
 * to PNG, and directory listing (POSIX dirent). This layer is deliberately
 * separate from the algorithm code: on the Alif target frames come from the
 * camera HAL, so only the pure ops in the other translation units are ported.
 */
#ifndef SOL_IMAGE_IO_HPP
#define SOL_IMAGE_IO_HPP

#include "Types.hpp"

#include <string>
#include <vector>

namespace sol {
namespace io {

    /** @brief Decode a JPEG/PNG file into an RGB (3-channel) image. */
    bool LoadRgb(const std::string& path, Image& out);

    /** @brief Write a single-channel 0/255 mask as a PNG. */
    bool SaveMaskPng(const std::string& path, const Image& mask);

    /**
     * @brief List the image files in @c dir (``.jpg``/``.jpeg``/``.png``),
     *        sorted by name. Returns file names only (not full paths).
     */
    std::vector<std::string> ListImageFiles(const std::string& dir);

    /** @brief Create @c dir if it does not already exist. */
    bool EnsureDir(const std::string& dir);

    /** @brief Join a directory and a file name with '/'. */
    std::string JoinPath(const std::string& dir, const std::string& name);

    /** @brief The file name without its extension. */
    std::string Stem(const std::string& name);

} /* namespace io */
} /* namespace sol */

#endif /* SOL_IMAGE_IO_HPP */
