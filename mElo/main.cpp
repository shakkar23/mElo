// mElo.cpp : Defines the entry point for the application.
//
#include "csv/csv.hpp"
#include "csv/json.hpp"

#include "main.h"
#include "mElo/construct_omega.hpp"
#include "mElo/update_c.hpp"
#include "mElo/predict.hpp"
#include "mElo/math.hpp"

#include <random>
#include <ranges>
#include <algorithm>


using namespace std;
enum {ROCK, PAPERS, SCISSORS, rps_players};
struct game {
	// indices into the matrix
	int player1;
	int player2;
	// 0 or 1
	int outcome;
};

std::vector<game> init_game_list();
std::vector<game> rps_games();
// this maps the name to the index in the matrix
std::unordered_map<std::string, int> player_map;



Matrix get_melo();
Matrix get_elo();
void get_player_map();
void save_melo(Matrix& MELO);
void save_elo(Matrix& MELO);
void save_player_map();



int tetrio_trainer() {
	get_player_map();

	std::vector<game> games_list = init_game_list();

	// write the player map to a csv file just in case there is new players in the data
	save_player_map();


	auto omega = construct_omega();
	/*
		r matrix, ELO matrix, n players x 1 (vector)
		c matrix, MELO matrix, n players x 2k MELO dimensions
		games_list, [(player i, player j, outcome)]

		for i, j, outcome in games_list:
			update_c(r[i], r[j], c[i], c[j], outcome)
	*/


	Matrix ELO = get_elo();

	Matrix MELO = get_melo();


	constexpr int batch_period = 10'000;

	for (int batch_n = 0; batch_n < 50; batch_n++) {

		Matrix MELO_copy = MELO;
		// elo copy

		int batch_counter = batch_period;

		for (auto&& [i, j, outcome] : games_list) {
			Column cA = MELO_copy.get_col(i);
			Column cB = MELO_copy.get_col(j);

			auto [elo_a, elo_b, melo_a, melo_b] = update_melo(ELO[i], ELO[j], cA, cB, outcome);

			MELO.set_col(i, MELO.get_col(i) + melo_a - cA);
			MELO.set_col(j, MELO.get_col(j) + melo_b - cB);

			// update the elo
			ELO[i] = elo_a;
			ELO[j] = elo_b;

			if (batch_counter == 0) {
				MELO_copy = MELO;
				batch_counter = batch_period;
			} else 
				batch_counter--;
		}

		std::cout << "saving data..." << std::endl;
		// save the melo and elo to csv files
		save_elo(ELO);
		save_melo(MELO);
		save_player_map();

		const auto example_player = "5f708143ea3d3a2b3abdfe23";

		Column prediction(player_map.size());
		for (auto&& [name, index] : player_map) {
			auto jth = MELO.get_col(index);
			auto ith = MELO.get_col(player_map[example_player]);
			prediction[index] = predict(ELO(0, player_map[example_player]), ELO(0, index), ith, jth);
		}
		// print prediction
		for (auto&& [name, index] : player_map) {
			std::cout << name << ": " << prediction[index] << std::endl;
		}

	}

	return 0;
}


int tetrio_viewer() {
	// init player map
	get_player_map();

	// get melo and elo
	Matrix MELO = get_melo();
	Matrix ELO = get_elo();


	// ask the user for a player name
	std::string prediction_player;
	std::cout << "Enter a player name to predict their win rate against one other player: ";
	std::cin >> prediction_player;

	// if the player is not in the map, ask the user for another name
	while (player_map.find(prediction_player) == player_map.end()) {
		std::cout << "Player not found, enter another name: ";
		std::cin >> prediction_player;
	}

	std::string opponent_player;
	std::cout << "Enter a player name to predict their win rate against one other player: ";
	std::cin >> opponent_player;
	
	while (player_map.find(opponent_player) == player_map.end()) {
		std::cout << "opponnet not found, enter another name: ";
		std::cin >> opponent_player;
	}

	Column prediction(player_map.size());

	auto jth = MELO.get_col(player_map[opponent_player]);
	auto ith = MELO.get_col(player_map[prediction_player]);

	std::cout << "PREDICTION:" << std::endl;
	std::cout << predict(ELO(0, player_map[prediction_player]), ELO(0, player_map[opponent_player]), ith, jth) << std::endl;

	return 0;
}

int tetrio_viewer_total() {


	// init player map
	get_player_map();

	// get melo and elo
	Matrix MELO = get_melo();
	Matrix ELO = get_elo();


	// ask the user for a player name
	std::string prediction_player;
	std::cout << "Enter a player name to predict their win rate against one other player: ";
	std::cin >> prediction_player;

	// if the player is not in the map, ask the user for another name
	while (player_map.find(prediction_player) == player_map.end()) {
		std::cout << "Player not found, enter another name: ";
		std::cin >> prediction_player;
	}


	Column prediction(player_map.size());
	for (auto&& [name, index] : player_map) {
		auto jth = MELO.get_col(index);
		auto ith = MELO.get_col(player_map[prediction_player]);
		prediction[index] = predict(ELO(0, player_map[prediction_player]), ELO(0, index), ith, jth);
	}

	// print prediction
	for (auto&& [name, index] : player_map) {
		std::cout << name << ": " << prediction[index] << std::endl;
	}

	return 0;
}

int rps() {
	// rock paper scissors

	auto games_list = rps_games();

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

	for (int batch_n = 0; batch_n < 8; batch_n++) {
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

	return 0;
}

int main() {
	return tetrio_trainer();
}

std::vector<game> init_game_list() {
	std::vector<game> games_list;

	// read the csv file
	csv::CSVReader reader("assets/games.csv");

	int i = 0;
	const int data_limit = 1000'000'000;

	for (auto&& row : reader) {
		i++;
		if (i > data_limit) {
			break;
		}
		std::string player1 = row[0].get<std::string>();
		std::string player2 = row[1].get<std::string>();

		if (player_map.find(player1) == player_map.end()) {
			player_map[player1] = player_map.size();
		}

		if (player_map.find(player2) == player_map.end()) {
			player_map[player2] = player_map.size();
		}

		games_list.push_back({ player_map[player1], player_map[player2], 1 });
		//games_list.push_back({ player_map[player2], player_map[player1], 0 });
	}

	return games_list;
}


std::vector<game> rps_games() {
	return {
		{ROCK, SCISSORS, 1},
		{ROCK, PAPERS, 0},
		{SCISSORS, PAPERS, 1},
		{SCISSORS, ROCK, 0},
		{PAPERS, ROCK, 1},
		{PAPERS, SCISSORS, 0}
	};
}


Matrix get_melo() {
	Matrix MELO(2*k, player_map.size());
	std::ifstream melo_file("assets/melo.csv");
	if (melo_file.is_open()) {
		csv::CSVReader reader(melo_file);
		int y = 0;
		for (auto&& row : reader) {
			for (int i = 0; i < row.size(); i++) {
				MELO(y, i) = row[i].get<float>();
			}
			y++;
		}
	} else {
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
	}

	return MELO;
}

Matrix get_elo() {
	Matrix ELO(1, player_map.size());
	std::ifstream elo_data_file("assets/elo_data.csv");
	if (elo_data_file.is_open()) {
		csv::CSVReader reader(elo_data_file);
		for (auto&& row : reader) {
			ELO(0, player_map[row[0].get<std::string>()]) = row[1].get<float>();
		}
	} else {
		// init elo with 2200
		for (int i = 0; i < ELO.width; i++) {
			ELO(0, i) = 2200;
		}
	}
	return ELO;
}

void get_player_map() {
	// read the csv file that contains the player map if it exists
	std::ifstream player_map_file("assets/player_map.csv");
	{
		if (!player_map_file.is_open()) {
			std::cout << "player_map.csv not found, creating new one" << std::endl;
			std::ofstream player_map_file("assets/player_map.csv");
		} else {
			csv::CSVReader reader(player_map_file);
			for (auto&& row : reader) {
				player_map[row[0].get<std::string>()] = row[1].get<int>();
			}
		}
	}
}

void save_melo(Matrix &MELO) {
	// save the melo and elo to csv files
	{
		std::ofstream melo_file_out("assets/melo.csv");
		melo_file_out.clear();
		auto writer = csv::make_csv_writer(melo_file_out);
		for (int y = 0; y < MELO.height; y++) {
			std::vector<float> row(MELO.width);
			for (int x = 0; x < MELO.width; x++) {
				row.push_back(MELO(y, x));
			}
			writer << row;
		}
		writer.flush();
		melo_file_out.close();
	}
}

void save_elo(Matrix& ELO) {
	{
		std::ofstream elo_data_file_out("assets/elo_data.csv");
		elo_data_file_out.clear();
		auto writer = csv::make_csv_writer(elo_data_file_out);
		for (auto&& [name, index] : player_map) {
			writer << std::array{ name, std::to_string(ELO(0, index)) };
		}
		writer.flush();
		elo_data_file_out.close();
	}
}

void save_player_map() {
	// write the player map to a csv file
	{
		std::ofstream player_map_file_out("assets/player_map.csv");
		auto writer = csv::make_csv_writer(player_map_file_out);
		for (auto&& [name, index] : player_map) {
			writer << std::array{ name, std::to_string(index) };
		}
	}
}

