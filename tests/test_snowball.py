import sys
sys.path.append('..')

from chaosNLP.knownledge_graph.relation_extraction.snowball import Snowball


def test_load_file():
    snowball = Snowball('en')
    res = snowball.load_labeled_corpus('./data/snowball_labeled_data.txt')
    assert len(res) > 0
    assert res[0].entity_a == 'Martin Schulz'
    assert res[0].type_a == 'PER'
    assert res[0].entity_b == 'Strasbourg'
    assert res[0].type_b == 'LOC'
    assert res[0].before == 'by euro deputy'
    assert res[0].middle.strip() == 'at the assembly in'
    assert res[0].after == 'constitute a serious'

    assert res[1].entity_a == 'Silvio Berlusconi'
    assert res[1].type_a == 'PER'
    assert res[1].entity_b == 'Italian'
    assert res[1].type_b == 'MSC'
    assert res[1].before == 'of the council'
    assert res[1].middle.strip() == 'and to'
    assert res[1].after == 'and'

    assert res[-2].entity_a == 'Japan Airlines'
    assert res[-2].type_a == 'ORG'
    assert res[-2].entity_b == 'Tokyo'
    assert res[-2].type_b == 'LOC'
    assert res[-2].before == 'its passengers to'
    assert res[-2].middle.strip() == 'flights to'
    assert res[-2].after == 'and beyond .'

    assert res[-1].entity_a == 'Wesson'
    assert res[-1].type_a == 'LOC'
    assert res[-1].entity_b == 'Clinton'
    assert res[-1].type_b == 'PER'
    assert res[-1].before == ''
    assert res[-1].middle.strip() == "'s agreement with President"
    assert res[-1].after == 'to begin marketing'
