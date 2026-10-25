dataset = sys.argv[1] if len(sys.argv) > 1 else "facebook"
    G = load_facebook() if dataset == "facebook" else load_youtube()