file(GLOB CPP_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

foreach(CPP_FILE ${CPP_FILES})
    get_filename_component(TARGET ${CPP_FILE} NAME_WE)
    add_executable(${TARGET} ${CPP_FILE})
    target_include_directories(${TARGET} PUBLIC ${ONNXRUNTIME_PATH}/include ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})
    target_link_libraries(${TARGET} PUBLIC ${ONNXRUNTIME_LIBS} ${OpenCV_LIBS} ${CUDA_LIBRARIES})
endforeach()
