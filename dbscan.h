/* Copyright (C) 2021 Imagry. All Rights Reserved.
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential.
 */

#pragma once

#include <cmath>
#include <vector>

#include <opencv2/core.hpp>
#include <nanoflann.hpp>

// Adapter for nanoflann to use std::vector<Point>
struct PointCloud {
    std::vector<cv::Point2f> pts;

    // Must return the number of data points
    inline size_t kdtree_get_point_count() const { return pts.size(); }

    // Returns the distance between the vector "p1[0:size-1]" and the data point
    // with index "idx_p2"
    inline float kdtree_distance(const float* p1, const size_t idx_p2,
                                 size_t /*size*/) const {
        const float d0 = p1[0] - pts[idx_p2].x;
        const float d1 = p1[1] - pts[idx_p2].y;
        return d0 * d0 + d1 * d1;
    }

    // Returns the dim'th component of the idx'th point in the class
    inline float kdtree_get_pt(const size_t idx, int dim) const {
        if (dim == 0)
            return pts[idx].x;
        else
            return pts[idx].y;
    }

    // Optional bounding-box computation
    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /*bb*/) const {
        return false;
    }
};

struct PointCloud4D {
    std::vector<cv::Vec4f> pts;

    // Must return the number of data points
    inline size_t kdtree_get_point_count() const { return pts.size(); }

    // Returns the distance between the vector "p1[0:size-1]" and the data point
    // with index "idx_p2"
    inline float kdtree_distance(const float* p1, const size_t idx_p2,
                                 size_t /*size*/) const {
        const float dx = p1[0] - pts[idx_p2][0];   // x
        const float dy = p1[1] - pts[idx_p2][1];   // y
        const float dvx = p1[2] - pts[idx_p2][2];  // vx
        const float dvy = p1[3] - pts[idx_p2][3];  // vy
        return dx * dx + dy * dy + dvx * dvx + dvy * dvy;
    }

    // Returns the dim'th component of the idx'th point in the class
    inline float kdtree_get_pt(const size_t idx, int dim) const {
        return pts[idx][dim];
    }

    // Optional bounding-box computation
    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /*bb*/) const {
        return false;
    }
};

typedef nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<float, PointCloud>, PointCloud, 2 /* dim */
    >
    KDTree;

typedef nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<float, PointCloud4D>, PointCloud4D, 4 /* dim */
    >
    KDTree4D;

std::vector<int> RegionQuery(const PointCloud& data, const KDTree& index,
                             int point_idx, float eps);

void ExpandCluster(const PointCloud& data, const KDTree& index, int point_idx,
                   float eps, int min_pts, std::vector<int>* cluster,
                   std::vector<bool>* visited, std::vector<bool>* noise);

std::vector<std::vector<int>> DBSCAN(const std::vector<cv::Point2f>& data,
                                     float eps, int minPts);

std::vector<int> RegionQuery4D(const PointCloud4D& data, const KDTree4D& index,
                               int point_idx, float eps);

void ExpandCluster4D(const PointCloud4D& data, const KDTree4D& index,
                     int point_idx, float eps, int min_pts,
                     std::vector<int>* cluster, std::vector<bool>* visited,
                     std::vector<bool>* noise);                               

std::vector<std::vector<int>> DBSCAN4D(const std::vector<cv::Vec4f>& points,
                                       float eps, int min_pts);