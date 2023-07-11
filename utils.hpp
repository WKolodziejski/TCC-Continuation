#ifndef UTILS_H
#define UTILS_H

#include "william.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>

#ifdef __cplusplus
extern "C" {
#endif

double distance(Correspondence &c);

double angle(Correspondence &c);

void print_latex(Stats &results, int d, int m, ofstream &latex);

void print_csv(Stats &results, int d, int m, ofstream &csv);

void print_cmd(Stats &results, const string &name, int d, int m, int i, int f,
               int t);

void stats_accumulate(Stats &stats_this, Stats &stats);

void stats_normalize(Stats &stats, int d);

void stats_percent(const vector<Stats> &stats);

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
