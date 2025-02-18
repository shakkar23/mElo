#pragma once

// ROCK PAPER SCISSORS

#include "common.hpp"

#include "../mElo/math.hpp"
#include "../mElo/update_c.hpp"
#include "../mElo/construct_omega.hpp"
#include "../mElo/predict.hpp"


#include <random>
#include <algorithm>
#include <ranges>
#include <vector>
#include <iostream>


namespace rps {

	enum { ROCK, PAPERS, SCISSORS, rps_players };
	
	inline std::vector<game> rps_games_list() {
		return {
			{ROCK, SCISSORS, 1},
			{ROCK, PAPERS, 0},
			{SCISSORS, PAPERS, 1},
			{SCISSORS, ROCK, 0},
			{PAPERS, ROCK, 1},
			{PAPERS, SCISSORS, 0}
		};
	}

	inline void trainer() {
		// rock paper scissors

		auto games_list = rps_games_list();

		auto omega = construct_omega();

		/*
			r matrix, ELO matrix, n players x 1 (vector)
			c matrix, MELO matrix, n players x 2k MELO dimensions
			games_list, [(player i, player j, outcome)]
			for i, j, outcome in games_list:
				update_c(r[i], r[j], c[i], c[j], outcome)
		*/

		Matrix ELO(1, rps_players);

		Matrix MELO(2 * k, rps_players);

		// init melo with random numbers between 0 and 1
		std::linear_congruential_engine<std::uint32_t, 0x5d588b65, 0x269ec3, 0> LCG(16);
		std::uniform_real_distribution<float> dist(0.0001, 1);
		for (int y = 0; y < MELO.height; y++) {
			for (int x = 0; x < MELO.width; x++) {
				MELO(y, x) = dist(LCG);
			}
			// softmax the row we just populated
			softmax(std::span(&MELO(y, 0), MELO.width));
		}


		constexpr int batch_period = 1000;

		for (int epoch = 0; epoch < 8; epoch++) {
			Matrix MELO_copy = MELO;
			// elo copy
			Matrix ELO_copy = ELO;

			int batch_counter = batch_period;

			std::ranges::shuffle(games_list, LCG);

			for (auto&& [i, j, outcome] : games_list) {
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
					batch_counter = batch_period;
				}

				batch_counter--;
			}
		}

		// print the melo
		for (int j = 0; j < MELO.height; j++) {
			for (int i = 0; i < MELO.width; i++) {
				std::cout << MELO(j, i) << " ";
			}
			std::cout << std::endl;
		}

		std::cout << std::endl;
		std::cout << std::endl;

		Matrix prediction(rps_players, rps_players);

		for (int i = 0; i < rps_players; i++) {
			for (int j = 0; j < rps_players; j++) {
				auto ith = MELO.get_col(i);
				auto jth = MELO.get_col(j);
				prediction(i, j) = predict(ELO(0, i), ELO(0, j), ith, jth);
			}
		}

		// print prediction
		for (int i = 0; i < rps_players; i++) {
			for (int j = 0; j < rps_players; j++) {
				std::cout << prediction(i, j) << " ";
			}
			std::cout << std::endl;
		}

	}

};



















