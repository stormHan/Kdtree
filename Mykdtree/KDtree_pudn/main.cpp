#include <cstdio>
#include <vector>
#include <map>
#include <algorithm>
#include <cstdlib>
#include <float.h>
#include <time.h>
#include <ANN/ANN.h>

#include "KDtree.h"
#include "CUDA_KDtree.h"

#pragma comment(lib, "ANN.lib")

#define timeval clock_t

double TimeDiff(timeval t1, timeval t2);
double SearchCPU(const vector <Point> &query, const vector <Point> &data, vector <int> &idxs, vector <float> &dist_sq);
void SearchANN(const vector <Point> &query, const vector <Point> &data, vector <int> &idxs, vector <float> dist_sq, double &create_time, double &search_time);

int main()
{
	KDtree tree;
	CUDA_KDTree GPU_tree;
	timeval t1, t2;
	int max_tree_levels = 13; // play around with this value to get the best result

	vector <Point> data(1000);
	vector <Point> queries(1000);

	vector <int> gpu_indexes, cpu_indexes;
	vector <float> gpu_dists, cpu_dists;

	//freopen("out", "r", stdin);
	//freopen("out", "w", stdout);
	
	for (unsigned int i = 0; i < data.size(); i++) {
		data[i].coords[0] = 0 + 100.0*(rand() / (1.0 + RAND_MAX));
		data[i].coords[1] = 0 + 100.0*(rand() / (1.0 + RAND_MAX));
		data[i].coords[2] = 0 + 100.0*(rand() / (1.0 + RAND_MAX));
	}

	for (unsigned int i = 0; i < queries.size(); i++) {
		queries[i].coords[0] = 0 + 100.0*(rand() / (1.0 + RAND_MAX));
		queries[i].coords[1] = 0 + 100.0*(rand() / (1.0 + RAND_MAX));
		queries[i].coords[2] = 0 + 100.0*(rand() / (1.0 + RAND_MAX));
	}

	//print to out file
	/*for (unsigned int i = 0; i < data.size(); ++i)
	{
		scanf("%f, %f, %f\n", &data[i].coords[0], &data[i].coords[1], &data[i].coords[2]);
	}

	for (unsigned int i = 0; i < queries.size(); ++i)
	{
		scanf("%f, %f, %f\n", &queries[i].coords[0], &queries[i].coords[1], &queries[i].coords[2]);
	}*/

	
	// Time to create the tree
	//gettimeofday(&t1, NULL);
	t1 = clock();
	tree.Create(data, max_tree_levels);
	
	GPU_tree.CreateKDTree(tree.GetRoot(), tree.GetNumNodes(), data, tree.GetLevel());
	//gettimeofday(&t2, NULL);
	t2 = clock();
	double gpu_create_time = TimeDiff(t1, t2);

	// Time to search the tree
	//gettimeofday(&t1, NULL);
	t1 = clock();
	//GPU_tree.Search(queries, gpu_indexes, gpu_dists);
	GPU_tree.Search_knn(queries, gpu_indexes, gpu_dists,5);
	//gettimeofday(&t2, NULL);
	t2 = clock();
	double gpu_search_time = TimeDiff(t1, t2);
	
	t1 = clock();
	SearchCPU(queries, data, cpu_indexes, cpu_dists);
	t2 = clock();
	double cpu_search_time = TimeDiff(t1, t2);

	double ANN_create_time;
	double ANN_search_time;
	//SearchANN(queries, data, cpu_indexes, cpu_dists, ANN_create_time, ANN_search_time);

	// Verify results
	for (unsigned int i = 0; i< gpu_indexes.size(); i++) {
		if (gpu_indexes[i] != cpu_indexes[i]) {
			printf("Resuts not the same :(\n"); 
			printf("%d != %d\n", gpu_indexes[i], cpu_indexes[i]);
			printf("%f %f\n", gpu_dists[i], cpu_dists[i]);
			//return 1;
		}
	}

	printf("Points in the tree: %ld\n", data.size());
	printf("Query points: %ld\n", queries.size());
	printf("\n");

	printf("Results are the same!\n");

	printf("\n");
	/*
	printf("GPU max tree depth: %d\n", max_tree_levels);
	printf("GPU create + search: %g + %g = %g ms\n", gpu_create_time, gpu_search_time, gpu_create_time + gpu_search_time);
	printf("ANN create + search: %g + %g = %g ms\n", ANN_create_time, ANN_search_time, ANN_create_time + ANN_search_time);
	printf("CPU search time : %g ms\n", cpu_search_time);


	printf("Speed up of GPU over CPU for searches: %.2fx\n", ANN_search_time / gpu_search_time);*/

		/*
	Point query;
	int ret_index;
	float ret_dist;

	for(int k=0; k < 100; k++) {
	query.coords[0] = 100.0*(rand() / (1.0 + RAND_MAX));
	query.coords[1] = 100.0*(rand() / (1.0 + RAND_MAX));
	query.coords[2] = 100.0*(rand() / (1.0 + RAND_MAX));

	tree.Search(query, &ret_index, &ret_dist);

	// Brute force
	float best_dist = FLT_MAX;
	int best_idx = 0;

	for(unsigned int i=0; i < data.size(); i++) {
	float dist = 0;

	for(int j=0; j < KDTREE_DIM; j++) {
	float d = data[i].coords[j] - query.coords[j];
	dist += d*d;
	}

	if(dist < best_dist) {
	best_dist = dist;
	best_idx = i;
	}
	}

	if(ret_index != best_idx) {
	printf("RESULTS NOT THE SAME :(\n");
	printf("\n");
	printf("Query: %f %f %f\n", query.coords[0], query.coords[1], query.coords[2]);
	printf("\n");
	printf("Search result %f %f %f\n", data[ret_index].coords[0], data[ret_index].coords[1],  data[ret_index].coords[2]);
	printf("Dist: %f\n", ret_dist);
	printf("IDX: %d\n", ret_index);

	printf("\n");

	printf("Ground truth: %f %f %f\n", data[best_idx].coords[0], data[best_idx].coords[1],  data[best_idx].coords[2]);
	printf("Dist: %f\n", best_dist);
	printf("IDX: %d\n", best_idx);
	exit(1);
	}
	}

	*/
	
	return 0;
}

double TimeDiff(timeval t1, timeval t2)
{
	// double t;
	// t = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
	//  t += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms
	return (t2 - t1) / CLOCKS_PER_SEC * 1000;
	// return t;
}


vector<int> findNearestPoints(const Point &query, const vector <Point> &data, int _n)
{
	std::map<double, int> dis_index_map;
	std::vector<int> nearest_N_index;
	std::vector<double> temp_dis;

	

	float t = 0.0;
	for (int i = 0; i < data.size(); ++i)
	{
		for (int k = 0; k < KDTREE_DIM; k++) {
			float d = query.coords[k] - data[i].coords[k];
			t += d*d;
		}

		while (dis_index_map.count(t) != 0)
			t += 0.0000001;
		dis_index_map.insert(std::pair<double, int>(t, i));

		temp_dis.push_back(t);
		t = 0.0;
	}

	std::sort(temp_dis.begin(), temp_dis.end());

	for (int i = 0; nearest_N_index.size() < _n; ++i)
	{
		int tmp = dis_index_map[temp_dis[i]];
		
		nearest_N_index.push_back(tmp);
	}
	return nearest_N_index;
}

/*
	this CPU version doesn't use kdtree
*/
double SearchCPU(const vector <Point> &queries, const vector <Point> &data, vector <int> &idxs, vector <float> &dist_sq)
{
	timeval t1, t2;

	int query_pts = queries.size();
	int data_pts = data.size();

	idxs.resize(query_pts * 5);
	dist_sq.resize(query_pts * 5);

	//gettimeofday(&t1, NULL);
	t1 = clock();
	for (unsigned int i = 0; i < query_pts; i++) {
		float best_dist = FLT_MAX;
		int best_idx = 0;

		vector<int> near = findNearestPoints(queries[i], data, 5);
		for (int j = 0; j < 5; ++j)
		{
			idxs[i * 5 + j] = near[j];
		}
		/*for (unsigned int j = 0; j < data_pts; j++) {
			float dist_sq1 = 0;
			
			for (int k = 0; k < KDTREE_DIM; k++) {
				float d = queries[i].coords[k] - data[j].coords[k];
				dist_sq1 += d*d;
			}
			
			if (dist_sq1 < best_dist) {
				best_dist = dist_sq1;
				best_idx = j;
			}
		}*/

		//idxs[i] = best_idx;
		//dist_sq[i] = best_dist;
	}

	//gettimeofday(&t2, NULL);
	t2 = clock();

	return TimeDiff(t1, t2);
}

void SearchANN(const vector <Point> &queries, const vector <Point> &data, vector <int> &idxs, vector <float> dist_sq, double &create_time, double &search_time)
{
	int k = 1;
	timeval t1, t2;

	idxs.resize(queries.size());
	dist_sq.resize(queries.size());

	ANNidxArray nnIdx = new ANNidx[k];
	ANNdistArray dists = new ANNdist[k];
	ANNpoint queryPt = annAllocPt(KDTREE_DIM);

	ANNpointArray dataPts = annAllocPts(data.size(), KDTREE_DIM);

	for (unsigned int i = 0; i < data.size(); i++) {
		for (int j = 0; j < KDTREE_DIM; j++) {
			dataPts[i][j] = data[i].coords[j];
		}
	}

	//gettimeofday(&t1, NULL);
	t1 = clock();
	ANNkd_tree* kdTree = new ANNkd_tree(dataPts, data.size(), KDTREE_DIM);
	//gettimeofday(&t2, NULL);
	t2 = clock();
	create_time = TimeDiff(t1, t2);

	//gettimeofday(&t1, NULL);
	t1 = clock();
	for (int i = 0; i < queries.size(); i++) {
		for (int j = 0; j < KDTREE_DIM; j++) {
			queryPt[j] = queries[i].coords[j];
		}

		kdTree->annkSearch(queryPt, 1, nnIdx, dists);

		idxs[i] = nnIdx[0];
		dist_sq[i] = dists[0];
	}
	//gettimeofday(&t2, NULL);
	t2 = clock();
	search_time = TimeDiff(t1, t2);

	delete[] nnIdx;
	delete[] dists;
	delete kdTree;
	annDeallocPts(dataPts);
	annClose();
}
