#include "BYTETracker.hpp"
#include <fstream>
#include <iostream>

using namespace std;

BYTETracker::BYTETracker(int frame_rate, int track_buffer)
{
	this->m_track_thresh = 0.5;
	this->m_high_thresh = 0.6;
	this->m_match_thresh = 0.8;

	this->m_frame_id = 0;
	this->m_max_time_lost = int(frame_rate / 30.0 * track_buffer);
	cout << "Init ByteTrack!" << endl;
}

BYTETracker::~BYTETracker()
{
}

vector<STrack> BYTETracker::update(const vector<arm::app::pg_object_detection::Detection>& objects)
{

	////////////////// Step 1: Get detections //////////////////
	this->m_frame_id++;
	vector<STrack> activated_stracks;
	vector<STrack> refind_stracks;
	vector<STrack> removed_stracks;
	vector<STrack> lost_stracks;
	vector<STrack> detections;
	vector<STrack> detections_low;

	vector<STrack> detections_cp;
	vector<STrack> tracked_stracks_swap;
	vector<STrack> resa, resb;
	vector<STrack> output_stracks;

	vector<STrack*> unconfirmed;
	vector<STrack*> tracked_stracks;
	vector<STrack*> strack_pool;
	vector<STrack*> r_tracked_stracks;

	if (objects.size() > 0)
	{
		for (int i = 0; i < (int)objects.size(); i++)
		{
			vector<float> tlbr_;
			tlbr_.resize(4);
			tlbr_[0] = objects[i].bbox.x;
			tlbr_[1] = objects[i].bbox.y;
			tlbr_[2] = objects[i].bbox.x + objects[i].bbox.w;
			tlbr_[3] = objects[i].bbox.y + objects[i].bbox.h;

			vector<float> pos_;
			pos_.resize(2);
			pos_[0] = objects[i].position.x;
			pos_[1] = objects[i].position.y;

			float score = objects[i].confidence;

			STrack strack(STrack::tlbr_to_tlwh(tlbr_), score, pos_);
			if (score >= this->m_track_thresh)
			{
				detections.push_back(strack);
			}
			else
			{
				detections_low.push_back(strack);
			}
			
		}
	}

	// Add newly detected tracklets to tracked_stracks
	for (int i = 0; i < (int)this->m_tracked_stracks.size(); i++)
	{
		if (!this->m_tracked_stracks[i].is_activated)
			unconfirmed.push_back(&this->m_tracked_stracks[i]);
		else
			tracked_stracks.push_back(&this->m_tracked_stracks[i]);
	}
	////////////////// Step 2: First association, with IoU //////////////////
	strack_pool = joint_stracks(tracked_stracks, this->m_lost_stracks);
	STrack::multi_predict(strack_pool, this->m_kalman_filter);

	vector<vector<float> > dists;
	int dist_size = 0, dist_size_size = 0;
	dists = iou_distance(strack_pool, detections, dist_size, dist_size_size);

	vector<vector<int> > matches;
	vector<int> u_track, u_detection;
	linear_assignment(dists, dist_size, dist_size_size, this->m_match_thresh, matches, u_track, u_detection);

	for (int i = 0; i < (int)matches.size(); i++)
	{
		STrack *track = strack_pool[matches[i][0]];
		STrack *det = &detections[matches[i][1]];
		if (track->state == TrackState::Tracked)
		{
			track->update(*det, this->m_frame_id);
			activated_stracks.push_back(*track);
		}
		else
		{
			track->re_activate(*det, this->m_frame_id, false);
			refind_stracks.push_back(*track);
		}
	}
	////////////////// Step 3: Second association, using low score dets //////////////////
	for (int i = 0; i < (int)u_detection.size(); i++)
	{
		detections_cp.push_back(detections[u_detection[i]]);
	}
	detections.clear();
	detections.assign(detections_low.begin(), detections_low.end());
	
	for (int i = 0; i < (int)u_track.size(); i++)
	{
		if (strack_pool[u_track[i]]->state == TrackState::Tracked)
		{
			r_tracked_stracks.push_back(strack_pool[u_track[i]]);
		}
	}

	dists.clear();
	dists = iou_distance(r_tracked_stracks, detections, dist_size, dist_size_size);

	matches.clear();
	u_track.clear();
	u_detection.clear();
	linear_assignment(dists, dist_size, dist_size_size, 0.5, matches, u_track, u_detection);

	for (int i = 0; i < (int)matches.size(); i++)
	{
		STrack *track = r_tracked_stracks[matches[i][0]];
		STrack *det = &detections[matches[i][1]];
		if (track->state == TrackState::Tracked)
		{
			track->update(*det, this->m_frame_id);
			activated_stracks.push_back(*track);
		}
		else
		{
			track->re_activate(*det, this->m_frame_id, false);
			refind_stracks.push_back(*track);
		}
	}

	for (int i = 0; i < (int)u_track.size(); i++)
	{
		STrack *track = r_tracked_stracks[u_track[i]];
		if (track->state != TrackState::Lost)
		{
			track->mark_lost();
			lost_stracks.push_back(*track);
		}
	}

	// Deal with unconfirmed tracks, usually tracks with only one beginning frame
	detections.clear();
	detections.assign(detections_cp.begin(), detections_cp.end());

	dists.clear();
	dists = iou_distance(unconfirmed, detections, dist_size, dist_size_size);

	matches.clear();
	vector<int> u_unconfirmed;
	u_detection.clear();
	linear_assignment(dists, dist_size, dist_size_size, 0.7, matches, u_unconfirmed, u_detection);

	for (int i = 0; i < (int)matches.size(); i++)
	{
		unconfirmed[matches[i][0]]->update(detections[matches[i][1]], this->m_frame_id);
		activated_stracks.push_back(*unconfirmed[matches[i][0]]);
	}

	for (int i = 0; i < (int)u_unconfirmed.size(); i++)
	{
		STrack *track = unconfirmed[u_unconfirmed[i]];
		track->mark_removed();
		removed_stracks.push_back(*track);
	}
	////////////////// Step 4: Init new stracks //////////////////
	for (int i = 0; i < (int)u_detection.size(); i++)
	{
		STrack *track = &detections[u_detection[i]];
		if (track->m_score < this->m_high_thresh)
			continue;
		track->activate(this->m_kalman_filter, this->m_frame_id);
		activated_stracks.push_back(*track);
	}
	////////////////// Step 5: Update state //////////////////
	for (int i = 0; i < (int)this->m_lost_stracks.size(); i++)
	{
		if (this->m_frame_id - this->m_lost_stracks[i].end_frame() > this->m_max_time_lost)
		{
			this->m_lost_stracks[i].mark_removed();
			removed_stracks.push_back(this->m_lost_stracks[i]);
		}
	}
	
	for (int i = 0; i < (int)this->m_tracked_stracks.size(); i++)
	{
		if (this->m_tracked_stracks[i].state == TrackState::Tracked)
		{
			tracked_stracks_swap.push_back(this->m_tracked_stracks[i]);
		}
	}
	this->m_tracked_stracks.clear();
	this->m_tracked_stracks.assign(tracked_stracks_swap.begin(), tracked_stracks_swap.end());

	this->m_tracked_stracks = joint_stracks(this->m_tracked_stracks, activated_stracks);
	this->m_tracked_stracks = joint_stracks(this->m_tracked_stracks, refind_stracks);

	//std::cout << activated_stracks.size() << std::endl;

	this->m_lost_stracks = sub_stracks(this->m_lost_stracks, this->m_tracked_stracks);
	for (int i = 0; i < (int)lost_stracks.size(); i++)
	{
		this->m_lost_stracks.push_back(lost_stracks[i]);
	}

	this->m_lost_stracks = sub_stracks(this->m_lost_stracks, this->m_removed_stracks);
	for (int i = 0; i < (int)removed_stracks.size(); i++)
	{
		this->m_removed_stracks.push_back(removed_stracks[i]);
	}
	
	remove_duplicate_stracks(resa, resb, this->m_tracked_stracks, this->m_lost_stracks);

	this->m_tracked_stracks.clear();
	this->m_tracked_stracks.assign(resa.begin(), resa.end());
	this->m_lost_stracks.clear();
	this->m_lost_stracks.assign(resb.begin(), resb.end());
	
	// for (int i = 0; i < this->m_tracked_stracks.size(); i++)
	// {
	// 	if (this->m_tracked_stracks[i].is_activated)
	// 	{
	// 		output_stracks.push_back(this->m_tracked_stracks[i]);
	// 	}
	// }
	// return all tracks
	output_stracks.insert(output_stracks.end(), this->m_tracked_stracks.begin(), this->m_tracked_stracks.end());
	output_stracks.insert(output_stracks.end(), this->m_lost_stracks.begin(), this->m_lost_stracks.end());
	return output_stracks;
}