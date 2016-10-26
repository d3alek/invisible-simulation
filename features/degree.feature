Feature: degree of polarization 
    Follows the sun, weaker at the sun and strongest at 90 degree shift from it	

    Scenario: zero at the sun
        Given the sun is at altitude zenith 
        and we look at altitude zenith 
        then degree is 0

    Scenario: large at the horizon
        Given the sun is at altitude zenith 
        and we look at altitude horizon 
        and we look at east
        then degree is 80

    Scenario: large at the zenith when sun is rising 
        Given the sun is at altitude horizon 
        and the sun is at east
        and we look at altitude zenith 
        then degree is 80

    Scenario: large at 90 degree from the sun
        Given the sun is at altitude horizon 
        and the sun is at east
        and we look at altitude horizon 
        and we look at south
        then degree is 80

    Scenario: zero at the antisun
        Given the sun is at altitude horizon 
        and the sun is at east
        and we look at altitude horizon 
        and we look at west
        then degree is 0

