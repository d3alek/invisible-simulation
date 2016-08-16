Feature: angle of polarization 
    Always perpendicular to line connecting the observed to the sun

    Scenario: angle goes from 0 to 90 on horizon at sunrise 1
        Given the sun is at the horizon 
        and the sun is at east 
        and we look at the horizon
        and we look at east
        then angle is horizontal

    Scenario: angle goes from 0 to 90 on horizon at sunrise 2
        Given the sun is at the horizon 
        and the sun is at east 
        and we look at the horizon
        and we look at south
        then angle is 60

    Scenario: angle goes from 0 to 90 on horizon at sunrise 3
        Given the sun is at the horizon 
        and the sun is at east 
        and we look at the horizon
        and we look at west 
        then angle is vertical

    Scenario: vertical at zenith on sunrise when looking south
        Given the sun is at the horizon 
        and the sun is at east 
        and we look at the zenith 
        and we look at south
        then angle is vertical 

    Scenario: horizontal at zenith on sunrise when looking east
        Given the sun is at the horizon 
        and the sun is at east 
        and we look at the zenith 
        and we look at east 
        then angle is horizontal

    Scenario: horizontal at horizon at noon when looking east
        Given the sun is at the zenith 
        and we look at the horizon
        and we look at east 
        then angle is horizontal

    Scenario: horizontal at horizon at noon when looking south
        Given the sun is at the zenith 
        and we look at the horizon
        and we look at south
        then angle is horizontal

    Scenario: horizontal at horizon at noon when looking west
        Given the sun is at the zenith 
        and we look at the horizon
        and we look at west
        then angle is horizontal

    Scenario: horizontal at horizon at noon when looking north
        Given the sun is at the zenith 
        and we look at the horizon
        and we look at north
        then angle is horizontal
