#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>

#include <Eigen/Dense>
#include "nlohmann/json.hpp"

#include "BYTETracker.hpp"

using namespace std;
using namespace arm::app::pg_object_detection;
using json = nlohmann::json;

void readData(const string& dataPath, vector<vector<Detection>>& detections)
{
    std::ifstream f(dataPath);
    json data = json::parse(f);

    for (const auto& item : data) {
        vector<Detection> tmpDetections;
        for (const auto& detection : item["detections"]) {
            vector<float> rawDetection;
            for (const auto& value : detection) {
                rawDetection.push_back(value);
            }
            Detection obj;
            obj.bbox.x = rawDetection[0];
            obj.bbox.y = rawDetection[1];
            obj.bbox.w = rawDetection[2];
            obj.bbox.h = rawDetection[3];
            obj.cls = rawDetection[5];
            obj.confidence = rawDetection[4];
            obj.position.x = rawDetection[6];
            obj.position.y = rawDetection[7];

            tmpDetections.push_back(obj);
        }
        detections.push_back(tmpDetections);
    }
}

void writeData(const string& outPath, vector<vector<Detection>>& detections, vector<vector<STrack>>& tracks)
{
    // Create an array to hold tracks objects
    json tracksArray = json::array();

    for (auto i = 0; i < detections.size(); i++) {
        
        vector<vector<float>> rawDetections;
        for (auto tmpDetection: detections[i]) {
            vector<float> rawDetection{
                tmpDetection.bbox.x,
                tmpDetection.bbox.y,
                tmpDetection.bbox.w,
                tmpDetection.bbox.h,
                tmpDetection.confidence,
                (float)tmpDetection.cls,
                tmpDetection.position.x,
                tmpDetection.position.y,
            };
            rawDetections.push_back(rawDetection);
        }

        json rawTracks = json::array();
        for (auto tmpTrack: tracks[i]) {
            json rawTrack;
            rawTrack["id"] = tmpTrack.track_id;
            rawTrack["state"] = tmpTrack.state;
            rawTrack["is_activated"] = tmpTrack.is_activated;
            rawTrack["tlwh"] = tmpTrack.tlwh;
            rawTrack["pos"] = tmpTrack.pos;
            rawTrack["score"] = tmpTrack.m_score;

            rawTracks.push_back(rawTrack);
        }

        json tmpDetections;
        tmpDetections["detections"] = rawDetections;
        tmpDetections["tracks"] = rawTracks;
        tracksArray.push_back(tmpDetections);
    }

    // Write JSON to file
    std::ofstream file(outPath);
    if (file.is_open()) {
        file << std::setw(4) << tracksArray << std::endl;
        file.close();
        std::cout << "JSON data has been written.\n";
    } else {
        cout << "Error opening file.\n";
    }
}

int main() {
    vector<vector<Detection>> detections;
    string dataPath = "/home/kyamkovyi/projects/pg/code/researches/tracker/artifacts/detections.json";
    string outPath = "/home/kyamkovyi/projects/pg/code/researches/tracker/artifacts/tracks_cpp.json";
    readData(dataPath, detections);

    BYTETracker tracker;
    vector<vector<STrack>> tracks;

    int num_frames = 0;
    int total_ms = 1;
    for (auto objects: detections) {
        num_frames ++;
        if (num_frames % 20 == 0)
        {
            cout << "Processing frame " << num_frames << " (" << num_frames * 1000000 / total_ms << " fps)" << endl;
        }
        auto start = chrono::system_clock::now();
        vector<STrack> output_stracks = tracker.update(objects);
        auto end = chrono::system_clock::now();
        total_ms = total_ms + chrono::duration_cast<chrono::microseconds>(end - start).count();
        tracks.push_back(output_stracks);
    }

    writeData(outPath, detections, tracks);

    return 0;
}
