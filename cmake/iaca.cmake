find_program(IACA_PROGRAM
	iaca
	HINTS ENV IACA_HOME
)

function(IACA_TARGET name srcs)
	if(IACA_PROGRAM)
		add_executable(exe_${name} EXCLUDE_FROM_ALL ${srcs})
		target_compile_definitions(exe_${name} PRIVATE "-D${name}=1")
		# enable all architecture features, not executed
		target_compile_options(exe_${name} PRIVATE -mavx2 -mavx512f -mavx512cd -mavx512dq)

		add_custom_target(${name}
			${IACA_PROGRAM} -arch SKX exe_${name}
			DEPENDS exe_${name}
			WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
		)
	else()
		message(STATUS "iaca not found, disabled all iaca targets")
	endif()
endfunction(IACA_TARGET)

