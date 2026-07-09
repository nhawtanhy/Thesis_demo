"""
Saved example cases shown on the Analysis page.

Structure: each entry is a CASE (one real D_test scenario, verified against
your actual dataset), containing a `results` list of per-method outcomes.
This lets the Analysis page show "base model used the deprecated API, but
DPO/GRPO didn't" (or "M1 retrieved the right context but the model still
used the deprecated API anyway") side by side for the same real input.

Fields per case:
    title               - short description (library: deprecated -> replacement)
    language            - for the syntax tag (usually "python")
    deprecated_snippet  - the code PREFIX shown to the model (from D_test's
                          "probing input new" field, verbatim — including
                          any whitespace artifacts from dataset construction,
                          since that's the literal input the model sees)

    results             - list of dicts, one per METHOD shown for this case
                          (not one per model_key — the same model_key can
                          appear twice, once with use_rag=False for M0 and
                          once with use_rag=True for M1/M2):
        model_key    - must match a key in model_backend.MODEL_REGISTRY
        method_label - override label for display (e.g. "M0 \u00b7 No RAG",
                       "M1 \u00b7 BM25+Rerank", "M2 \u00b7 Intent-Extended")
                       falls back to the model's registry label if omitted
        use_rag      - whether RAG context should be toggled on when this
                       case is sent live via "Try live" (default False)
        context_used - RAG context shown for this specific result row
                       (only relevant/shown if use_rag=True)
        completion   - PLACEHOLDER until you run this case live and paste
                       the actual model output here
        outcome      - "deprecated" | "replacement" | "other" | "untested"
                       "untested" renders as a neutral gray badge and
                       should be used until you've verified the real output
                       — do NOT mark deprecated/replacement until confirmed
    notes               - optional free text shown under the case

ALL SEVEN CASES BELOW ARE REAL, VERIFIED D_test ENTRIES (pulled directly
from your dataset, with real deprecated/replacement APIs and real
reference lines) but the per-model `completion`/`outcome` values are
PLACEHOLDERS. Run each case live in the Playground, confirm the actual
output, then replace the placeholder fields before your defense.
"""

EXAMPLES = [
    # ── Case 1: weight-update comparison (M0 vs M3 vs M4) ──────────────────
    {
        "title": "numpy.alltrue \u2192 numpy.all (rms_from_ivar)",
        "language": "python",
        "deprecated_snippet": (
            "def rms_from_ivar(ivar,parea=None,cylindrical=True):\n"
            "    \"\"\"\n"
            "    Return rms noise for each pixel in a map in physical units\n"
            "    (uK-arcmin) given a map of the inverse variance per pixel.\n"
            "    Optionally, provide a map of the pixel area.\n"
            "    \"\"\"\n"
            "    if parea is None:\n"
            "        shape,wcs = ivar.shape, ivar.wcs\n"
            "        parea = psizemap(shape,wcs) if cylindrical else enmap.pixsizemap(shape,wcs)\n"
            "    with np.errstate(divide='ignore', invalid='ignore',over='ignore'):\n"
            "        var = (1./ivar)\n"
            "    var[ivar<=0] = 0\n"
            "    assertnp."
        ),
        "results": [
            {
                "model_key": "deepseek-coder-1.3b-instruct",
                "method_label": "M0 \u00b7 Base (no RAG)",
                "use_rag": False,
                "rag_method": "none",
                "completion": "[PLACEHOLDER \u2014 run live and paste actual output]",
                "outcome": "untested",
            },
            {
                "model_key": "deepseek-coder-1.3b-dpo",
                "method_label": "M3 \u00b7 DPO",
                "use_rag": False,
                "rag_method": "none",
                "completion": "[PLACEHOLDER \u2014 run live and paste actual output]",
                "outcome": "untested",
            },
            {
                "model_key": "deepseek-coder-1.3b-grpo",
                "method_label": "M4 \u00b7 GRPO",
                "use_rag": False,
                "rag_method": "none",
                "completion": "[PLACEHOLDER \u2014 run live and paste actual output]",
                "outcome": "untested",
            },
        ],
        "notes": (
            "Real D_test case. Deprecated: numpy.alltrue \u2192 Replacement: numpy.all. "
            "Reference: assertnp.all(np.isfinite(var)) \u2014 note the missing space "
            "before 'np' is an artifact of dataset construction, kept verbatim "
            "since it's the literal input the model sees."
        ),
    },

    # ── Case 2: M1 (RAG) vs M0, same model ──────────────────────────────────
    {
        "title": "numpy.product \u2192 numpy.prod (make_permutation_code)",
        "language": "python",
        "deprecated_snippet": (
            "def make_permutation_code(V, vshape, pshape, t_in, t_out, array_name):\n"
            "    _, _, shifts = get_permutation_to_line_elements(V)\n"
            "    shift = shifts[0]\n"
            "    if shift != (0,):\n"
            "    ndof=numpy."
        ),
        "results": [
            {
                "model_key": "deepseek-coder-1.3b-instruct",
                "method_label": "M0 \u00b7 Base (no RAG)",
                "use_rag": False,
                "rag_method": "none",
                "completion": "[PLACEHOLDER \u2014 run live and paste actual output]",
                "outcome": "untested",
            },
            {
                "model_key": "deepseek-coder-1.3b-instruct",
                "method_label": "M1 \u00b7 BM25 + Rerank RAG",
                "use_rag": True,
                "rag_method": "m1",
                "context_used": (
                    "Replacement API: numpy.prod\n"
                    "Deprecated: ['numpy.product']"
                ),
                "completion": "[PLACEHOLDER \u2014 run live and paste actual output]",
                "outcome": "untested",
            },
        ],
        "notes": (
            "Real D_test case. Deprecated: numpy.product \u2192 Replacement: numpy.prod. "
            "Reference: ndof=numpy.prod(vshape). Use this case to show M1's "
            "retrieved context alongside the base model's unaided output \u2014 "
            "and note if DUR stays flat despite correct context being retrieved "
            "(a key finding from the thesis worth highlighting live)."
        ),
    },

    # ── Case 3: M2 vs M1, same model (intent-extended vs code-only) ────────
    {
        "title": "numpy.product \u2192 numpy.prod (get_required_memory_size)",
        "language": "python",
        "deprecated_snippet": (
            "    def get_required_memory_size(self, view_model):\n"
            "        # type: (FourierSpectrumModel) -> dict\n"
            "        \"\"\"\n"
            "        Return the required memory to run this algorithm.\n"
            "        \"\"\"\n"
            "        fs_input_index = self.load_entity_by_gid(view_model.input_data)\n"
            "        returnnumpy."
        ),
        "results": [
            {
                "model_key": "deepseek-coder-1.3b-instruct",
                "method_label": "M1 \u00b7 BM25 + Rerank RAG",
                "use_rag": True,
                "rag_method": "m1",
                "context_used": (
                    "Replacement API: numpy.prod\n"
                    "Deprecated: ['numpy.product']"
                ),
                "completion": "[PLACEHOLDER \u2014 run live and paste actual output]",
                "outcome": "untested",
            },
            {
                "model_key": "deepseek-coder-1.3b-instruct",
                "method_label": "M2 \u00b7 Intent-Extended RAG",
                "use_rag": True,
                "rag_method": "m2",
                "intention": "Calculate and return the memory size needed to execute the algorithm.",
                "context_used": (
                    "Intent: \"Calculate and return the memory size needed to "
                    "execute the algorithm.\"\n"
                    "Replacement API: numpy.prod\n"
                    "Deprecated: ['numpy.product']"
                ),
                "completion": "[PLACEHOLDER \u2014 run live and paste actual output]",
                "outcome": "untested",
            },
        ],
        "notes": (
            "Real D_test case, real intent field from the dataset: "
            "\"Calculate and return the memory size needed to execute the "
            "algorithm.\" Use this case to illustrate M2's intent-extended "
            "query \u2014 note this pair only differs in the retrieval query, "
            "same underlying deprecated/replacement pair as Case 2."
        ),
    },

    # ── Case 4: GRPO vs DPO divergence ──────────────────────────────────────
    {
        "title": "numpy.alltrue \u2192 numpy.all (test_encapsulation)",
        "language": "python",
        "deprecated_snippet": (
            "    def test_encapsulation(self):\n"
            "        \"\"\"Test the matrix encapsulation.\"\"\"\n\n"
            "        # check that a sparse matrix will be converted to a CSC format\n"
            "        expected_matrix = numpy.array([\n"
            "            [1.0, 2.0, 3.0],\n"
            "            [0.0, 1.0, 4.0],\n"
            "            [0.0, 0.0, 1.0]])\n\n"
            "        matrix = SparseTermSimilarityMatrix(scipy.sparse.csc_matrix(expected_matrix)).matrix\n"
            "        self.assertTrue(isinstance(matrix, scipy.sparse.csc_matrix))\n"
            "        self.assertTrue(numpy."
        ),
        "results": [
            {
                "model_key": "deepseek-coder-1.3b-dpo",
                "method_label": "M3 \u00b7 DPO",
                "use_rag": False,
                "rag_method": "none",
                "completion": "[PLACEHOLDER \u2014 run live and paste actual output]",
                "outcome": "untested",
            },
            {
                "model_key": "deepseek-coder-1.3b-grpo",
                "method_label": "M4 \u00b7 GRPO",
                "use_rag": False,
                "rag_method": "none",
                "completion": "[PLACEHOLDER \u2014 run live and paste actual output]",
                "outcome": "untested",
            },
        ],
        "notes": (
            "Real D_test case \u2014 a test-function scenario (per your thesis's "
            "\"test function\" special case: the deprecated API's use should be "
            "judged on the operation being tested, not the test wrapper itself). "
            "Good candidate for showing GRPO vs DPO divergence if their outputs differ."
        ),
    },

    # ── Case 5: long/complex context, tests retrieval + generation together ─
    {
        "title": "seaborn.tsplot \u2192 seaborn.lineplot (gen_adv_plots)",
        "language": "python",
        "deprecated_snippet": (
            "def gen_adv_plots(df_image, ood=False):\n"
            "    print(f\"Generating calibration plot with ood={ood}\")\n"
            "    df = df_image[df_image[\"OOD\"]==ood]\n"
            "    df_pixel = df_image_to_pixels(df, keys=[\"Target\", \"Mu\", \"Sigma\", \"Epsilon\"])\n"
            "    df_pixel[\"Error\"] = np.abs(df_pixel[\"Mu\"] - df_pixel[\"Target\"])\n"
            "    df_pixel[\"Entropy\"] = 0.5*np.log(2*np.pi*np.exp(1.)*(df_pixel[\"Sigma\"]**2))\n"
            "    print (df_pixel[df_pixel[\"Method\"]==\"Laplace\"].head() )\n"
            "    df_pixel[\"Entropy\"].mask(df_pixel[\"Method\"] == \"Laplace\", "
            "np.log(2*df_pixel[\"Sigma\"]*np.exp(1.)), inplace=True)\n"
            "    print (df_pixel[df_pixel[\"Method\"]==\"Laplace\"].head() )\n\n"
            "    ### Plot epsilon vs error per method\n"
            "    df = df_pixel.groupby([df_pixel.index, \"Method\", \"Model Path\", \"Epsilon\"]).mean().reset_index()\n"
            "    df_by_method = df_pixel.groupby([\"Method\", \"Model Path\", \"Epsilon\"]).mean().reset_index()\n"
            "    sns."
        ),
        "results": [
            {
                "model_key": "deepseek-coder-1.3b-instruct",
                "method_label": "M0 \u00b7 Base (no RAG)",
                "use_rag": False,
                "rag_method": "none",
                "completion": "[PLACEHOLDER \u2014 run live and paste actual output]",
                "outcome": "untested",
            },
            {
                "model_key": "deepseek-coder-1.3b-instruct",
                "method_label": "M2 \u00b7 Intent-Extended RAG",
                "use_rag": True,
                "rag_method": "m2",
                "intention": "Generate plots for adversarial robustness analysis.",
                "context_used": (
                    "Intent: \"Generate plots for adversarial robustness analysis.\"\n"
                    "Replacement API: seaborn.lineplot\n"
                    "Deprecated: ['seaborn.tsplot']"
                ),
                "completion": "[PLACEHOLDER \u2014 run live and paste actual output]",
                "outcome": "untested",
            },
        ],
        "notes": (
            "Real D_test case with a long, realistic function body (research-code "
            "style, not a toy example) \u2014 good stress test for whether retrieval "
            "and generation still work cleanly on messier, real-world input rather "
            "than clean textbook snippets."
        ),
    },

    # ── Case 6: different library, base-model weight-update case ───────────
    {
        "title": "sklearn.mixture.GMM \u2192 GaussianMixture",
        "language": "python",
        "deprecated_snippet": (
            "ll = defaultdict(dict)\n"
            "acc = defaultdict(lambda: defaultdict(dict))\n"
            "adjMI = defaultdict(lambda: defaultdict(dict))\n"
            "km = kmeans(random_state=5)\n"
            "gmm="
        ),
        "results": [
            {
                "model_key": "codegen-2b-mono",
                "method_label": "M0 \u00b7 Base (no RAG)",
                "use_rag": False,
                "rag_method": "none",
                "completion": "[PLACEHOLDER \u2014 run live and paste actual output]",
                "outcome": "untested",
            },
            {
                "model_key": "codegen-2b-dpo",
                "method_label": "M3 \u00b7 DPO",
                "use_rag": False,
                "rag_method": "none",
                "completion": "[PLACEHOLDER \u2014 run live and paste actual output]",
                "outcome": "untested",
            },
            {
                "model_key": "codegen-2b-grpo",
                "method_label": "M4 \u00b7 GRPO",
                "use_rag": False,
                "rag_method": "none",
                "completion": "[PLACEHOLDER \u2014 run live and paste actual output]",
                "outcome": "untested",
            },
        ],
        "notes": (
            "Real D_test case, different library (sklearn) and a different "
            "deprecation shape \u2014 the alias IS the deprecated class name itself "
            "('GMM' \u2192 'sklearn.mixture.GaussianMixture'), not a module-path "
            "rename. Good for showing the CodeGen (base model) pipeline on a "
            "non-numpy example."
        ),
    },

    # ── Case 7: deprecated API absent from visible snippet — model must
    #            pick the right constructor among several plausible options ──
    {
        "title": "sklearn.preprocessing.Imputer \u2192 sklearn.impute.SimpleImputer",
        "language": "python",
        "deprecated_snippet": (
            "def imputers():\n"
            "return[IterativeImputer(tol=0.1),KNNImputer(),"
        ),
        "results": [
            {
                "model_key": "deepseek-coder-1.3b-instruct",
                "method_label": "M0 \u00b7 Base (no RAG)",
                "use_rag": False,
                "rag_method": "none",
                "completion": "[PLACEHOLDER \u2014 run live and paste actual output]",
                "outcome": "untested",
            },
            {
                "model_key": "deepseek-coder-1.3b-dpo",
                "method_label": "M3 \u00b7 DPO",
                "use_rag": False,
                "rag_method": "none",
                "completion": "[PLACEHOLDER \u2014 run live and paste actual output]",
                "outcome": "untested",
            },
            {
                "model_key": "deepseek-coder-1.3b-grpo",
                "method_label": "M4 \u00b7 GRPO",
                "use_rag": False,
                "rag_method": "none",
                "completion": "[PLACEHOLDER \u2014 run live and paste actual output]",
                "outcome": "untested",
            },
        ],
        "notes": (
            "Real D_test case \u2014 structurally different from the others: the "
            "deprecated API (sklearn.preprocessing.Imputer) never appears in "
            "the visible snippet at all. The model must complete a list "
            "literal alongside two OTHER real imputer classes "
            "(IterativeImputer, KNNImputer) and correctly choose "
            "SimpleImputer as the third item, rather than reproducing a "
            "deprecated call directly. Good test of whether the model "
            "recognizes the right class among several plausible siblings, "
            "not just recalling a single API name."
        ),
    },
]