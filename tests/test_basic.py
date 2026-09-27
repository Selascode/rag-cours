from rag_cours import main


def test_main(capsys):
    main()
    captured = capsys.readouterr()
    assert "Hello from rag-cours!" in captured.out
