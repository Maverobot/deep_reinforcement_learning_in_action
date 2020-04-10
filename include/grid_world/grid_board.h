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
  GridBoard(uint32_t size = 4) : size_(size){};

  void addPiece(std::string name, char code, uint32_t row_idx, uint32_t col_idx) {
    if (!findPiece(name)) {
      pieces_.push_back({std::move(name), code, row_idx, col_idx});
    } else {
      std::ostringstream oss;
      oss << "addPiece failed. Piece with name " << name << " already exists.";
      throw std::runtime_error(oss.str());
    }
  }

  auto getPiecePos(const std::string& name) {
    if (auto piece = findPiece(name)) {
      return std::pair<uint32_t, uint32_t>(piece->get().row_idx, piece->get().col_idx);
    } else {
      std::ostringstream oss;
      oss << "getPiecePos failed. Piece with name " << name << " does not exist.";
      throw std::runtime_error(oss.str());
    }
  }

  void movePiece(std::string name, uint32_t row_idx_delta, uint32_t col_idx_delta) {
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
  bool isOnBoard(std::pair<uint32_t, uint32_t> pos) { return isOnBoard(pos.first, pos.second); }

  bool isOnBoard(uint32_t row_idx, uint32_t col_idx) {
    return row_idx >= 0 && row_idx < size_ && col_idx >= 0 && col_idx < size_;
  }

 private:
  struct BoardPiece {
    std::string name;
    char code;
    uint32_t row_idx;
    uint32_t col_idx;
  };

  uint32_t size_;
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
