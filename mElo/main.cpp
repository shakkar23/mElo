// mElo.cpp : Defines the entry point for the application.
//
#include "csv/csv.hpp"
#include "csv/json.hpp"

#include "main.h"

#include "all_games/arbitrary.hpp"
#include "all_games/RPSLS.hpp"
#include "all_games/RPS.hpp"

#include <random>
#include <ranges>
#include <algorithm>


using namespace std;


int main() {
	rpsls::trainer();
}

