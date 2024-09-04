#include <iostream>
#include "dbscan.h"

int main() {
    // Create a DBSCAN object
    DBSCAN dbscan;

    // Load data from a file or generate it programmatically
    // ...

    // Set the parameters for DBSCAN
    dbscan.setEpsilon(0.5);
    dbscan.setMinPoints(5);

    // Run DBSCAN on the data
    dbscan.run();

    // Get the clusters and noise points
    std::vector<std::vector<int>> clusters = dbscan.getClusters();
    std::vector<int> noisePoints = dbscan.getNoisePoints();

    // Print the results
    std::cout << "Number of clusters: " << clusters.size() << std::endl;
    for (int i = 0; i < clusters.size(); i++) {
        std::cout << "Cluster " << i << ": ";
        for (int j = 0; j < clusters[i].size(); j++) {
            std::cout << clusters[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "Number of noise points: " << noisePoints.size() << std::endl;
    for (int i = 0; i < noisePoints.size(); i++) {
        std::cout << "Noise point " << i << ": " << noisePoints[i] << std::endl;
    }

    return 0;
}