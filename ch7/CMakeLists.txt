find_package(OpenCV REQUIRED)
add_executable(orb_cv orb_cv.cpp)

add_custom_target(cp_img)
message("CMAKE_COMMAND: ${CMAKE_COMMAND}")
message("CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
message("CMAKE_BINARY_DIR: ${CMAKE_BINARY_DIR}")

add_custom_command(
  TARGET cp_img PRE_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
    ${CMAKE_CURRENT_SOURCE_DIR}/*.png
    ${CMAKE_BINARY_DIR}/ch7/
)
target_link_libraries(orb_cv ${OpenCV_LIBS})
add_dependencies(orb_cv cp_img)

