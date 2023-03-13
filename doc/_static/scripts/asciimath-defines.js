window.MathJax.startup = {
    ready: () => {
        AM = MathJax.InputJax.AsciiMath.AM;
        AM.newsymbol({ input: "ell", tag: "mi", output: "\u2113", tex: "ell", ttype: AM.TOKEN.CONST });
        AM.newsymbol({ input: "||", tag: "mi", output: "\u2225", tex: "Vert", ttype: AM.TOKEN.CONST });
        AM.newsymbol({ input: "triangleq", tag: "mo", output: "\u225C", tex: "triangleq", ttype: AM.TOKEN.CONST });
        MathJax.startup.defaultReady();
    }
};
