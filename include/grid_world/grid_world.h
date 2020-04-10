#pragma once

#include <grid_board.h>

namespace drl_in_action::grid_world {
class GridWorld {
 public:
  enum class Action { up, down, left, right };
  using Reward = float;

  GridWorld(uint32_t size = 4) : board_(size) {
    // Initial board state
    board_.addPiece("Player", 'P', 0, 0);
    board_.addPiece("Wall", 'W', 1, 2);
    board_.addPiece("Pit", '-', 2, 2);
    board_.addPiece("Goal", '+', 3, 3);
  };

  auto step(Action action, bool verbose = false) {
    if (status_ != Status::ongoing) {
      throw std::runtime_error("cannot step a finished game.");
    }
    if (verbose) {
      printAction(action);
    }
    auto [row_idx_delta, col_idx_delta] = mapAction(action);
    if (isValid(row_idx_delta, col_idx_delta)) {
      board_.movePiece("Player", row_idx_delta, col_idx_delta);
    } else {
      std::cout << "Invalid move\n";
    }
    return std::pair(evaluate(), board_.state());
  }

  void display() const {
    switch (status_) {
      case Status::ongoing:
        board_.display();
        break;
      case Status::win:
        std::cout << "You won the game by reaching the goal\n";
        break;
      case Status::loss:
        std::cout << "You loss the game by falling in the pit\n";
        break;
    }
  }

  auto state() const { return board_.state(); }

  bool over() const { return status_ != Status::ongoing; }

 private:
  GridBoard board_;

  enum class Status { ongoing, loss, win };
  Status status_{Status::ongoing};

  void printAction(Action action) const {
    switch (action) {
      case Action::up:
        std::cout << "Move up\n";
        break;
      case Action::down:
        std::cout << "Move down\n";
        break;
      case Action::left:
        std::cout << "Move left\n";
        break;
      case Action::right:
        std::cout << "Move right\n";
        break;
    }
  }

  bool isValid(uint32_t row_idx_delta, uint32_t col_idx_delta) {
    auto new_player_pos =
        std::pair<uint32_t, uint32_t>(board_.getPiecePos("Player").first + row_idx_delta,
                                      board_.getPiecePos("Player").second + col_idx_delta);
    return new_player_pos != board_.getPiecePos("Wall") && board_.isOnBoard(new_player_pos);
  }

  std::pair<uint32_t, uint32_t> mapAction(Action action) {
    uint32_t row_idx_delta = 0;
    uint32_t col_idx_delta = 0;
    switch (action) {
      case Action::up:
        row_idx_delta = -1;
        break;
      case Action::down:
        row_idx_delta = 1;
        break;
      case Action::left:
        col_idx_delta = -1;
        break;
      case Action::right:
        col_idx_delta = 1;
        break;
    }
    return {row_idx_delta, col_idx_delta};
  }

  Reward evaluate() {
    if (board_.getPiecePos("Player") == board_.getPiecePos("Pit")) {
      status_ = Status::loss;
      return -10;
    } else if (board_.getPiecePos("Player") == board_.getPiecePos("Goal")) {
      status_ = Status::win;
      return 10;
    } else {
      return -1;
    }
    return Reward();
  }
};
}  // namespace drl_in_action::grid_world
