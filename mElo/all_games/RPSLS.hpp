#pragma once



#include "common.hpp"
#include "../mElo/predict.hpp"
#include "../mElo/construct_omega.hpp"
#include "../mElo/update_c.hpp"
#include "../mElo/math.hpp"


#include <algorithm>
#include <ranges>
#include <random>

// rock paper scissors lizard spock
namespace rpsls {

	enum { ROCK, PAPERS, SCISSORS, LIZARD, SPOCK, rpsls_players };
	inline std::vector<game> rpsls_games_list() {
		return {
			{ROCK,		SCISSORS,	1},
			{ROCK,		LIZARD,		1},
			{ROCK,		PAPERS,		0},
			{ROCK,		SPOCK,		0},

			{PAPERS,	ROCK,		1},
			{PAPERS,	SPOCK,		1},
			{PAPERS,	SCISSORS,	0},
			{PAPERS,	LIZARD,		0},
			
			{SCISSORS,	PAPERS,		1},
			{SCISSORS,	LIZARD,		1},
			{SCISSORS,	ROCK,		0},
			{SCISSORS,	SPOCK,		0},
			
			{LIZARD,	SPOCK,		1},
			{LIZARD,	PAPERS,		1},
			{LIZARD,	ROCK,		0},
			{LIZARD,	SCISSORS,	0},
			
			{SPOCK,		ROCK,		1},
			{SPOCK,		SCISSORS,	1},
			{SPOCK,		LIZARD,		0},
			{SPOCK,		PAPERS,		0}
		};

	}

	inline void trainer() {
		// rock paper scissors lizard spock
		auto games_list = rpsls_games_list();
		auto omega = construct_omega();
		/*
			r matrix, ELO matrix, n players x 1 (vector)
			c matrix, MELO matrix, n players x 2k MELO dimensions
			games_list, [(player i, player j, outcome)]
			for i, j, outcome in games_list:
				update_c(r[i], r[j], c[i], c[j], outcome)
		*/
		Matrix ELO(1, rpsls_players);
		std::ranges::fill_n(ELO.data.begin(), rpsls_players, 2200);

		Matrix MELO(2 * k, rpsls_players);
		// init melo with random numbers between 0 and 1
		std::linear_congruential_engine<std::uint32_t, 0x5d588b65, 0x269ec3, 0> LCG(16);
		constexpr float epsilon = std::numeric_limits<float>::epsilon();
		std::uniform_real_distribution<float> dist(epsilon, 1 - epsilon);
		for (int y = 0; y < MELO.height; y++) {
			for (int x = 0; x < MELO.width; x++) {
				MELO(y, x) = dist(LCG);
			}
			// softmax the row we just populated
			softmax(std::span(&MELO(y, 0), MELO.width));
		}

		for (int epoch = 0; epoch < 50; epoch++) {
			Matrix MELO_copy = MELO;
			// elo copy
			Matrix ELO_copy = ELO;
			int batch_counter = 10;
			std::ranges::shuffle(games_list, LCG);
			for (auto&& [i, j, outcome] : games_list ) {
				Column cA = MELO_copy.get_col(i);
				Column cB = MELO_copy.get_col(j);
				auto [elo_a, elo_b, melo_a, melo_b] = update_melo(ELO_copy[i], ELO_copy[j], cA, cB, outcome);
				MELO.set_col(i, MELO.get_col(i) + melo_a - cA);
				MELO.set_col(j, MELO.get_col(j) + melo_b - cB);
				// update the elo
				ELO[i] = elo_a;
				ELO[j] = elo_b;
				if (batch_counter == 0) {
					MELO_copy = MELO;
					ELO_copy = ELO;
					batch_counter = 10'000;
				}
				batch_counter--;
			}
		}

		// print the melo
		std::cout << "melo:" << std::endl;
		for (int j = 0; j < MELO.height; j++) {
			for (int i = 0; i < MELO.width; i++) {
				std::cout << MELO(j, i) << " ";
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;


		// print the prediction
		std::cout << "prediction:" << std::endl;
		for (int i = 0; i < rpsls_players; i++) {
			for (int j = 0; j < rpsls_players; j++) {
				auto ith = MELO.get_col(i);
				auto jth = MELO.get_col(j);
				
				// to limit the output to 2 decimal places
				std::cout << std::fixed << std::setprecision(2);
				std::cout << predict(ELO[i], ELO[j], ith, jth) << " ";
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;


		// actual results
		std::cout << "actual results:" << std::endl;
		for (int i = 0; i < rpsls_players; i++) {
			for (int j = 0; j < rpsls_players; j++) {
				bool win = false;

				if (i == SPOCK) {
					if (j == SCISSORS || j == ROCK) win = true;

				} else if (i == SCISSORS) {
					if (j == PAPERS || j == LIZARD) win = true;
				
				} else if (i == PAPERS) {
					if (j == ROCK || j == SPOCK) win = true;
				
				} else if (i == LIZARD) {
					if (j == SPOCK || j == PAPERS) win = true;
				
				} else if (i == ROCK) {
					if (j == LIZARD || j == SCISSORS) win = true;
				}

				std::cout << std::fixed << std::setprecision(2);
				if (i == j) 
					std::cout << Numeric(0.5) << " ";
				else if(win)
					std::cout << Numeric(1) << " ";
				else 
					std::cout << Numeric(0) << " ";


			}
			std::cout << std::endl;
		}


		// print the elo
		std::cout << "elo:" << std::endl;
		for (int i = 0; i < ELO.width; i++) {
			std::cout << ELO(0, i) << " ";
		}
		std::cout << std::endl;

	}
};
