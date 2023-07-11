#ifndef UTILS_H
#define UTILS_H

#include "solution.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>

#ifdef __cplusplus
extern "C" {
#endif

double distance(Correspondence &c);

double angle(Correspondence &c);

string detectName(int d);

string matchName(int m);

string describeName(int d);

string modeName(int m);

string estimateName(int e);

string formatNameSolution(Detect detect_type, Match match_type,
                          Estimate estimate_type, int frame);

string formatName(const string &prefix, int frame);

string formatNameCluster(const string &prefix, const string &ver, int frame,
                         int k);

#ifdef __cplusplus
}
#endif

#endif  // UTILS_H
