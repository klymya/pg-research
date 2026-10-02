/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Command-line driver for the Sign-of-Life pipeline. Reads a folder of JPEG
 * frames, runs the pipeline in temporal order, and writes per-frame mask images
 * plus a text report to the output folder.
 *
 * Usage:
 *   sign_of_life <input_dir> <output_dir> [--min-area N] [--sol-area N]
 *                [--shadow-filter]
 */
#include "ImageIo.hpp"
#include "Pipeline.hpp"
#include "Types.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

    using sol::Blob;

    struct Options
    {
        std::string inputDir;
        std::string outputDir;
        int minArea       = -1; /* -1 = keep the config default */
        int areaThreshold = -1;
        bool shadowFilter = false; /* off by default, as on target */
    };

    void PrintUsage(const char* program)
    {
        std::fprintf(stderr,
                     "Usage: %s <input_dir> <output_dir> [--min-area N] [--sol-area N]\n"
                     "                [--shadow-filter]\n"
                     "  <input_dir>       folder of JPEG/PNG frames (processed in name order)\n"
                     "  <output_dir>      folder for the *_raw.png / *_morph.png masks and\n"
                     "                    the sign_of_life.txt report\n"
                     "  --min-area N      smallest kept foreground object in px (default 150)\n"
                     "  --sol-area N      Sign-of-Life area threshold in px (default 500)\n"
                     "  --shadow-filter   enable the HSV shadow filter (default off, as on\n"
                     "                    target); when off its background reference is\n"
                     "                    never allocated\n"
                     "  --no-shadow-veto  accepted for compatibility; the filter is off by\n"
                     "                    default\n",
                     program);
    }

    bool ParseArgs(int argc, char** argv, Options& options)
    {
        std::vector<std::string> positionals;
        for (int i = 1; i < argc; ++i)
        {
            const std::string arg = argv[i];
            if (arg == "--min-area" && i + 1 < argc)
            {
                options.minArea = std::atoi(argv[++i]);
            }
            else if (arg == "--sol-area" && i + 1 < argc)
            {
                options.areaThreshold = std::atoi(argv[++i]);
            }
            else if (arg == "--shadow-filter")
            {
                options.shadowFilter = true;
            }
            else if (arg == "--no-shadow-veto")
            {
                options.shadowFilter = false;
            }
            else if (arg == "-h" || arg == "--help")
            {
                return false;
            }
            else if (!arg.empty() && arg[0] == '-')
            {
                std::fprintf(stderr, "Unknown option: %s\n", arg.c_str());
                return false;
            }
            else
            {
                positionals.push_back(arg);
            }
        }
        if (positionals.size() != 2)
        {
            return false;
        }
        options.inputDir  = positionals[0];
        options.outputDir = positionals[1];
        return true;
    }

    std::string ObjectsToString(const std::vector<Blob>& blobs)
    {
        std::string text;
        for (size_t i = 0; i < blobs.size(); ++i)
        {
            const Blob& b = blobs[i];
            char buffer[64];
            std::snprintf(buffer, sizeof(buffer), "%d@%d,%d,%d,%d", b.area, b.x, b.y,
                          b.width, b.height);
            if (i != 0)
            {
                text += ';';
            }
            text += buffer;
        }
        return text;
    }

} /* namespace */

int main(int argc, char** argv)
{
    Options options;
    if (!ParseArgs(argc, argv, options))
    {
        PrintUsage(argv[0]);
        return 1;
    }

    sol::PipelineConfig config;
    if (options.minArea >= 0)
    {
        config.minArea = options.minArea;
    }
    if (options.areaThreshold >= 0)
    {
        config.areaThreshold = options.areaThreshold;
    }
    config.shadowEnabled = options.shadowFilter;

    const std::vector<std::string> frames = sol::io::ListImageFiles(options.inputDir);
    if (frames.empty())
    {
        std::fprintf(stderr, "No image files found in: %s\n", options.inputDir.c_str());
        return 1;
    }
    if (!sol::io::EnsureDir(options.outputDir))
    {
        std::fprintf(stderr, "Cannot create output directory: %s\n", options.outputDir.c_str());
        return 1;
    }

    const std::string reportPath = sol::io::JoinPath(options.outputDir, "sign_of_life.txt");
    std::ofstream report(reportPath);
    if (!report.is_open())
    {
        std::fprintf(stderr, "Cannot write report: %s\n", reportPath.c_str());
        return 1;
    }
    report << "# Sign-of-Life report\n";
    report << "# min_area=" << config.minArea
           << " area_threshold=" << config.areaThreshold
           << " shadow_veto=" << (config.shadowEnabled ? 1 : 0) << "\n";
    report << "# columns: frame  sign_of_life  num_objects  objects[area@x,y,w,h;...]\n";

    sol::Pipeline pipeline(config);
    int written = 0;
    int solFrames = 0;

    for (const std::string& name : frames)
    {
        sol::Image rgb;
        if (!sol::io::LoadRgb(sol::io::JoinPath(options.inputDir, name), rgb))
        {
            std::fprintf(stderr, "Skipping unreadable frame: %s\n", name.c_str());
            continue;
        }

        const sol::FrameResult result = pipeline.Process(rgb);

        const std::string stem = sol::io::Stem(name);
        sol::io::SaveMaskPng(sol::io::JoinPath(options.outputDir, stem + "_raw.png"),
                             result.rawMask);
        sol::io::SaveMaskPng(sol::io::JoinPath(options.outputDir, stem + "_morph.png"),
                             result.filteredMask);
        ++written;
        solFrames += result.signOfLife ? 1 : 0;

        report << name << "  " << (result.signOfLife ? 1 : 0) << "  "
               << result.blobs.size() << "  " << ObjectsToString(result.blobs) << "\n";
    }

    report << "# summary: " << written << " frames, " << solFrames
           << " sign-of-life\n";
    report.close();

    std::printf("Processed %d frames -> %s (%d Sign-of-Life, area >= %d px)\n", written,
                options.outputDir.c_str(), solFrames, config.areaThreshold);
    return 0;
}
