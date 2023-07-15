set(TARGETS single_image_model_inference fixed_batch_image_model_inference dynamic_batch_image_model_inference)

foreach(TARGET ${TARGETS})
    add_executable(${TARGET} ${TARGET}.cpp)
    target_include_directories(${TARGET} PUBLIC ${ONNXRUNTIME_PATH}/include)
    target_link_libraries(${TARGET} PUBLIC ${ONNXRUNTIME_LIB} ${OpenCV_LIBS})
endforeach()
