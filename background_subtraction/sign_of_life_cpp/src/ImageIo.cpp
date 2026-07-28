/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Implementation of the host I/O layer. See ImageIo.hpp.
 */
#include "ImageIo.hpp"

#include <algorithm>
#include <cctype>

#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace sol {
namespace io {

    namespace {

        std::string ToLower(std::string s)
        {
            std::transform(s.begin(), s.end(), s.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            return s;
        }

        bool HasImageExtension(const std::string& name)
        {
            const std::string lower = ToLower(name);
            const char* exts[] = {".jpg", ".jpeg", ".png"};
            for (const char* ext : exts)
            {
                const std::string e(ext);
                if (lower.size() >= e.size() &&
                    lower.compare(lower.size() - e.size(), e.size(), e) == 0)
                {
                    return true;
                }
            }
            return false;
        }

    } /* namespace */

    bool LoadRgb(const std::string& path, Image& out)
    {
        int w = 0;
        int h = 0;
        int channels = 0;
        unsigned char* pixels = stbi_load(path.c_str(), &w, &h, &channels, 3);
        if (pixels == nullptr)
        {
            return false;
        }
        out = Image(w, h, 3);
        std::copy(pixels, pixels + static_cast<size_t>(w) * h * 3, out.data.begin());
        stbi_image_free(pixels);
        return true;
    }

    bool SaveMaskPng(const std::string& path, const Image& mask)
    {
        return stbi_write_png(path.c_str(), mask.width, mask.height, 1, mask.data.data(),
                              mask.width) != 0;
    }

    std::vector<std::string> ListImageFiles(const std::string& dir)
    {
        std::vector<std::string> files;
        DIR* handle = opendir(dir.c_str());
        if (handle == nullptr)
        {
            return files;
        }
        for (dirent* entry = readdir(handle); entry != nullptr; entry = readdir(handle))
        {
            const std::string name = entry->d_name;
            if (HasImageExtension(name))
            {
                files.push_back(name);
            }
        }
        closedir(handle);
        std::sort(files.begin(), files.end());
        return files;
    }

    bool EnsureDir(const std::string& dir)
    {
        struct stat info;
        if (stat(dir.c_str(), &info) == 0)
        {
            return (info.st_mode & S_IFDIR) != 0;
        }
        return mkdir(dir.c_str(), 0755) == 0;
    }

    std::string JoinPath(const std::string& dir, const std::string& name)
    {
        if (dir.empty())
        {
            return name;
        }
        if (dir.back() == '/')
        {
            return dir + name;
        }
        return dir + "/" + name;
    }

    std::string Stem(const std::string& name)
    {
        const size_t slash = name.find_last_of('/');
        const std::string base = (slash == std::string::npos) ? name : name.substr(slash + 1);
        const size_t dot = base.find_last_of('.');
        return (dot == std::string::npos) ? base : base.substr(0, dot);
    }

} /* namespace io */
} /* namespace sol */
