#pragma once

#include <where.h>

#include <fort.hpp>

#include <algorithm>
#include <functional>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace drl_in_action::grid_world {
class GridBoard {
 public:
  struct BoardPiece {
    std::string name;
    char code;
    size_t row_idx;
    size_t col_idx;
  };

  GridBoard(size_t size = 4) : size_(size){};

  void addPiece(BoardPiece piece) {
    if (!findPiece(piece.name)) {
      pieces_.push_back(piece);
    } else {
      std::ostringstream oss;
      oss << "addPiece failed. Piece with name " << piece.name << " already exists.";
      throw std::runtime_error(oss.str());
    }
  }

  void addPiece(std::string name, char code, size_t row_idx, size_t col_idx) {
    addPiece({std::move(name), code, row_idx, col_idx});
  }

  auto getPiecePos(const std::string& name) {
    if (auto piece = findPiece(name)) {
      return std::pair<size_t, size_t>(piece->get().row_idx, piece->get().col_idx);
    } else {
      std::ostringstream oss;
      oss << "getPiecePos failed. Piece with name " << name << " does not exist.";
      throw std::runtime_error(oss.str());
    }
  }

  void movePiece(std::string name, size_t row_idx_delta, size_t col_idx_delta) {
    if (auto piece = findPiece(name)) {
      piece->get().row_idx += row_idx_delta;
      piece->get().col_idx += col_idx_delta;
    } else {
      std::ostringstream oss;
      oss << "movePiece failed. Piece with name " << name << " does not exist.";
      throw std::runtime_error(oss.str());
    }
  }

  std::vector<std::vector<float>> state() const {
    std::vector<std::vector<float>> state;
    for (const auto& piece : pieces_) {
      std::vector<float> state_p(size_ * size_, 0);
      state_p.at(piece.row_idx * size_ + piece.col_idx) = 1;
      state.push_back(state_p);
    }
    return state;
  }

  void display() const {
    fort::char_table table;
    for (size_t row_idx = 0; row_idx < size_; row_idx++) {
      for (size_t col_idx = 0; col_idx < size_; col_idx++) {
        const auto& piece = utils::where(pieces_, [row_idx, col_idx](const auto& piece) {
          return piece.row_idx == row_idx && piece.col_idx == col_idx;
        });

        if (piece.size() == 0) {
          table << " ";
        } else if (piece.size() == 1) {
          table << piece.at(0).get().code;
        } else {
          throw std::runtime_error("more than one pieces_ at a pos");
        }
      }
      table << fort::endr << fort::separator;
    }
    std::cout << table.to_string() << std::endl;
  };

  constexpr bool isOnBoard(std::pair<size_t, size_t> pos) const noexcept {
    return isOnBoard(pos.first, pos.second);
  }

  constexpr bool isOnBoard(size_t row_idx, size_t col_idx) const noexcept {
    return row_idx >= 0 && row_idx < size_ && col_idx >= 0 && col_idx < size_;
  }

  constexpr size_t size() const noexcept { return size_; }

 private:
  size_t size_;
  std::vector<BoardPiece> pieces_;

  std::optional<std::reference_wrapper<BoardPiece>> findPiece(const std::string& name) {
    if (auto iter = std::find_if(pieces_.begin(), pieces_.end(),
                                 [&name](auto& piece) { return piece.name == name; });
        iter != pieces_.end()) {
      return std::ref(*iter);
    }
    return {};
  }
};
}  // namespace drl_in_action::grid_world
