#pragma once

#include <grid_board.h>

#include <random>
#include <set>

namespace drl_in_action::grid_world {
class GridWorld {
 public:
  enum class InitMode { fixed, player_random, shuffle };
  enum class Action { up, down, left, right };
  using Reward = float;

  GridWorld(InitMode init_mode = InitMode::shuffle, size_t size = 4) : board_(size) {
    ValidBoardCreator creator(board_.size());
    switch (init_mode) {
      case InitMode::fixed:
        board_.addPiece("Player", 'P', 0, 3);
        board_.addPiece("Goal", '+', 0, 0);
        board_.addPiece("Wall", 'W', 1, 1);
        board_.addPiece("Pit", '-', 0, 1);
        break;
      case InitMode::player_random:
        creator.addPiece({"Goal", '+', 0, 0});
        creator.addPiece({"Wall", 'W', 1, 1});
        creator.addPiece({"Pit", '-', 0, 1});
        creator.addRandomPiece("Player", 'P');
        for (const auto& piece : creator.pieces()) {
          board_.addPiece(piece);
        }
        break;
      case InitMode::shuffle:
        creator.addRandomPiece("Goal", '+');
        creator.addRandomPiece("Player", 'P');
        creator.addRandomPiece("Wall", 'W');
        creator.addRandomPiece("Pit", '-');
        for (const auto& piece : creator.pieces()) {
          board_.addPiece(piece);
        }
        break;
    }
  };

  auto step(Action action, bool verbose = false) {
    if (status_ != Status::ongoing) {
      throw std::runtime_error("cannot step a finished game.");
    }
    if (verbose) {
      printAction(action);
    }
    auto [row_idx_delta, col_idx_delta] = mapAction(action);
    if (isPlayerMoveValid(row_idx_delta, col_idx_delta)) {
      board_.movePiece("Player", row_idx_delta, col_idx_delta);
    } else if (verbose) {
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
  bool win() const { return status_ == Status::win; }

 private:
  GridBoard board_;

  enum class Status { ongoing, loss, win };
  Status status_{Status::ongoing};

  class ValidBoardCreator {
   public:
    explicit ValidBoardCreator(size_t size) : dist_int_(0, size - 1){};

    bool addPiece(GridBoard::BoardPiece new_piece) {
      // Check if new piece overlapps with old ones
      if (std::find_if(pieces_.cbegin(), pieces_.cend(), [&new_piece](const auto& piece) {
            return new_piece.col_idx == piece.col_idx && new_piece.row_idx == piece.row_idx;
          }) != pieces_.cend()) {
        return false;
      }
      pieces_.push_back(std::move(new_piece));
      return true;
    }

    void addRandomPiece(std::string name, char code) {
      for (;;) {
        size_t rand_row_idx = dist_int_(rd_);
        size_t rand_col_idx = dist_int_(rd_);
        if (std::find_if(pieces_.cbegin(), pieces_.cend(),
                         [rand_row_idx, rand_col_idx](const auto& piece) {
                           return rand_col_idx == piece.col_idx && rand_row_idx == piece.row_idx;
                         }) == pieces_.cend()) {
          pieces_.push_back({name, code, rand_row_idx, rand_col_idx});
          return;
        }
      }
    };

    const std::vector<GridBoard::BoardPiece>& pieces() const { return pieces_; }

   private:
    std::vector<GridBoard::BoardPiece> pieces_;
    std::random_device rd_;
    std::uniform_int_distribution<int> dist_int_;
  };

  static void printAction(Action action) {
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

  static bool isBoardValid(const std::vector<GridBoard::BoardPiece>& pieces) {
    // Check overlapping
    std::set<std::pair<size_t, size_t>> positions;
    for (const auto& piece : pieces) {
      auto [_, inserted] = positions.insert(std::pair(piece.row_idx, piece.col_idx));
      if (!inserted) {
        // Overlapping detected
        return false;
      }
    }
    return true;
  }

  bool isPlayerMoveValid(size_t row_idx_delta, size_t col_idx_delta) {
    auto new_player_pos =
        std::pair<size_t, size_t>(board_.getPiecePos("Player").first + row_idx_delta,
                                  board_.getPiecePos("Player").second + col_idx_delta);
    return new_player_pos != board_.getPiecePos("Wall") && board_.isOnBoard(new_player_pos);
  }

  std::pair<size_t, size_t> mapAction(Action action) {
    size_t row_idx_delta = 0;
    size_t col_idx_delta = 0;
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
