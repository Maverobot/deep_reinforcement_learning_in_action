#include <grid_world/grid_world.h>

#include <iostream>

using drl_in_action::grid_world::GridWorld;

int main(int argc, char* argv[]) {
  GridWorld world;
  world.display();
  std::cout << world.step(GridWorld::Action::right).first << "\n";
  world.display();
  std::cout << world.step(GridWorld::Action::down).first << "\n";
  world.display();
  std::cout << world.step(GridWorld::Action::right).first << "\n";
  world.display();
  std::cout << world.step(GridWorld::Action::down).first << "\n";
  world.display();
  std::cout << world.step(GridWorld::Action::down).first << "\n";
  world.display();
  std::cout << world.step(GridWorld::Action::right).first << "\n";
  world.display();
  std::cout << world.step(GridWorld::Action::right).first << "\n";
  world.display();
  return 0;
}
