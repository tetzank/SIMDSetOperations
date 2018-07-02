find_program(IACA_PROGRAM
	iaca
	HINTS ENV IACA_HOME
)
find_path(IACA_INCLUDE_DIR
	iacaMarks.h
	HINTS ENV IACA_HOME
)
if(IACA_PROGRAM AND IACA_INCLUDE_DIR)
	set(IACA_FOUND TRUE)
endif()

function(IACA_TARGET name srcs)
	add_executable(exe_${name} EXCLUDE_FROM_ALL ${srcs})
	target_compile_definitions(exe_${name} PRIVATE "-DIACA=1" "-D${name}=1")
	# enable all architecture features, not executed
	target_compile_options(exe_${name} PRIVATE -mavx2 -mavx512f -mavx512cd -mavx512dq)
	target_include_directories(exe_${name} PRIVATE "${IACA_INCLUDE_DIR}")

	add_custom_target(${name}
		${IACA_PROGRAM} -arch SKX exe_${name}
		DEPENDS exe_${name}
		WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
	)
endfunction(IACA_TARGET)
