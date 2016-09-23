Feature: angle of polarization 
    Always perpendicular to line connecting the observed to the sun

    Scenario: angle is horizonal when looking at sun
        Given the sun is at the horizon 
        and the sun is at east 
        and we look at the horizon
        and we look at east
        then angle is horizontal

    Scenario: angle is horizontal when looking at anti-sun
        Given the sun is at the horizon 
        and the sun is at east 
        and we look at the horizon
        and we look at west 
        then angle is horizontal

    Scenario: vertical at horizon on sunrize when looking south
        Given the sun is at the horizon 
        and the sun is at east 
        and we look at the horizon
        and we look at south
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
        and we look at south
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
