Feature: calculating gamma on the horizon
	This is the angular distance between the observed pointing and the sun

	Background: assume sun and observed pointing are at the horizon
		Given we look at the horizon
		and the sun is at the horizon

    Scenario: zero gamma
        Given the sun is at east 
        and we look at east 
        then gamma is 0

    Scenario: middle gamma
        Given the sun is at east 
        and we look at south 
        then gamma is 90

    Scenario: middle gamma 2
        Given the sun is at east 
        and we look at north 
        then gamma is 90

    Scenario: maximum gamma
        Given the sun is at east 
        and we look at west
        then gamma is 180

