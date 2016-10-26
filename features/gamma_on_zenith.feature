Feature: harder gamma tests
	This is the angular distance between the observed pointing and the sun

    Scenario: zero gamma
        Given the sun is at altitude zenith 
        and we look at altitude zenith 
        then gamma is 0

    Scenario: medium gamma
        Given the sun is at altitude zenith 
        and we look at east 
        then gamma is 90 

    Scenario: medium gamma 2
        Given the sun is at altitude zenith 
        and we look at west 
        then gamma is 90

    Scenario: medium gamma 3
        Given the sun is at altitude zenith 
        and we look at north 
        then gamma is 90

    Scenario: medium gamma 4
        Given the sun is at altitude zenith 
        and we look at south 
        then gamma is 90
