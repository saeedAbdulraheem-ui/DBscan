/* Copyright (C) 2021 Imagry. All Rights Reserved.
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential.
 */
#include "dbscan.h"

#include <vector>


std::vector<int> RegionQuery(const PointCloud& data, const KDTree& index,
                             int point_idx, float eps) {
    std::vector<int> neighbors;
    std::vector<nanoflann::ResultItem<unsigned int, float>> ret_matches;
    nanoflann::SearchParameters params;
    const float query_pt[2] = {data.pts[point_idx].x, data.pts[point_idx].y};
    const size_t nMatches =
        index.radiusSearch(&query_pt[0], eps * eps, ret_matches, params);
    for (size_t i = 0; i < nMatches; i++) {
        neighbors.push_back(ret_matches[i].first);  // Changed .id to .first
    }
    return neighbors;
}

void ExpandCluster(const PointCloud& data, const KDTree& index, int point_idx,
                   float eps, int min_pts, std::vector<int>* cluster,
                   std::vector<bool>* visited, std::vector<bool>* noise) {
    std::vector<int> neighbors = RegionQuery(data, index, point_idx, eps);

    if (neighbors.size() < min_pts) {
        // The point is a noise point
        noise->at(point_idx) = true;
    } else {
        // Add point to cluster
        cluster->push_back(point_idx);

        for (int i = 0; i < neighbors.size(); i++) {
            int next_point_idx = neighbors[i];

            if (!visited->at(next_point_idx)) {
                visited->at(next_point_idx) = true;

                std::vector<int> next_point_neighbors =
                    RegionQuery(data, index, next_point_idx, eps);

                if (next_point_neighbors.size() >= min_pts) {
                    neighbors.insert(neighbors.end(),
                                     next_point_neighbors.begin(),
                                     next_point_neighbors.end());
                }
            }

            // If next_point_idx is not yet member of any cluster, add it to
            // this cluster
            if (!noise->at(next_point_idx)) {
                cluster->push_back(next_point_idx);
            }
        }
    }
}

std::vector<int> RegionQuery4D(const PointCloud4D& data, const KDTree4D& index,
                               int point_idx, float eps) {
    std::vector<int> neighbors;
    std::vector<nanoflann::ResultItem<unsigned int, float>> ret_matches;
    nanoflann::SearchParameters params;
    const float query_pt[4] = {data.pts[point_idx][0], data.pts[point_idx][1],
                               data.pts[point_idx][2], data.pts[point_idx][3]};
    const size_t nMatches =
        index.radiusSearch(&query_pt[0], eps * eps, ret_matches, params);
    for (size_t i = 0; i < nMatches; i++) {
        neighbors.push_back(ret_matches[i].first);
    }
    return neighbors;
}

void ExpandCluster4D(const PointCloud4D& data, const KDTree4D& index,
                     int point_idx, float eps, int min_pts,
                     std::vector<int>* cluster, std::vector<bool>* visited,
                     std::vector<bool>* noise) {
    std::vector<int> neighbors = RegionQuery4D(data, index, point_idx, eps);

    if (neighbors.size() < min_pts) {
        // The point is a noise point
        noise->at(point_idx) = true;
    } else {
        // Add point to cluster
        cluster->push_back(point_idx);

        for (int i = 0; i < neighbors.size(); i++) {
            int next_point_idx = neighbors[i];

            if (!visited->at(next_point_idx)) {
                visited->at(next_point_idx) = true;

                std::vector<int> next_point_neighbors =
                    RegionQuery4D(data, index, next_point_idx, eps);

                if (next_point_neighbors.size() >= min_pts) {
                    neighbors.insert(neighbors.end(),
                                     next_point_neighbors.begin(),
                                     next_point_neighbors.end());
                }
            }

            // If next_point_idx is not yet member of any cluster, add it to
            // this cluster
            if (!noise->at(next_point_idx)) {
                cluster->push_back(next_point_idx);
            }
        }
    }
}

std::vector<std::vector<int>> DBSCAN(const std::vector<cv::Point2f>& points,
                                     float eps, int min_pts) {
    PointCloud cloud;
    cloud.pts = points;
    KDTree index(2, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(3));
    index.buildIndex();

    std::vector<bool> visited(points.size(), false);
    std::vector<bool> noise(points.size(), false);
    std::vector<std::vector<int>> clusters;

    for (int i = 0; i < points.size(); i++) {
        if (!visited[i]) {
            visited[i] = true;
            std::vector<int> cluster;
            ExpandCluster(cloud, index, i, eps, min_pts, &cluster, &visited,
                          &noise);
            if (!cluster.empty()) {
                clusters.push_back(cluster);
            }
        }
    }

    return clusters;
}

std::vector<std::vector<int>> DBSCAN4D(const std::vector<cv::Vec4f>& points,
                                       float eps, int min_pts) {
    PointCloud4D cloud;
    cloud.pts = points;
    KDTree4D index(4, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(3));
    index.buildIndex();

    std::vector<bool> visited(points.size(), false);
    std::vector<bool> noise(points.size(), false);
    std::vector<std::vector<int>> clusters;

    for (int i = 0; i < points.size(); i++) {
        if (!visited[i]) {
            visited[i] = true;
            std::vector<int> cluster;
            ExpandCluster4D(cloud, index, i, eps, min_pts, &cluster, &visited,
                            &noise);
            if (!cluster.empty()) {
                clusters.push_back(cluster);
            }
        }
    }

    return clusters;
}

