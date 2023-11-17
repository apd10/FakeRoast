
def add_roast_args(parser):
    roast_args = parser.add_argument_group('roast')
    roast_args.add_argument('--roast-module-limit-size', type=int, default=-1,
                        help='do not roast belwo this size')
    roast_args.add_argument('--roast-init-std', type=float, default=0.05,
                        help='std dev')
    roast_args.add_argument('--roast-compression', type=float, default=0,
                        help='-log10(sparsity)')
    roast_args.add_argument('--roast-scaler-mode', type=str, default="v1",
                        help='scaler mode')
    roast_args.add_argument('--roast-seed', type=int, default=123321,
                        help='roast seed')
    roast_args.add_argument('--roast-mapper', type=str, default="pareto",
                        help='scaler mode')
