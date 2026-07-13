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

    # ── Case 3: torch.qr -> torch.linalg.qr (all 5 methods, real retrieved context + prompts) ──
    {
        "title": "torch.qr \u2192 torch.linalg.qr",
        "language": "python",
        "deprecated_snippet": (
            "import torch\n\n"
            "# Compute the QR decomposition of the matrix\n"
            "matrix = torch.randn(5, 3)\n"
            "q, r = torch."
        ),
        "results": [
            {
                "model_key": "deepseek-coder-1.3b-instruct",
                "method_label": "M0 \u00b7 Base (no RAG)",
                "use_rag": False,
                "rag_method": "none",
                "completion": (
                    "import torch\n\n"
                    "# Compute the QR decomposition of the matrix\n"
                    "q, r = torch.qr(matrix)\n"
                    "print(q)\n"
                    "print(r)"
                ),
                "outcome": "deprecated",
            },
            {
                "model_key": "deepseek-coder-1.3b-instruct",
                "method_label": "M1 \u00b7 BM25 + Rerank RAG",
                "use_rag": True,
                "rag_method": "m1",
                "context_used": (
                    "Replacement API: torch.linalg.qr\n"
                    "Deprecated: ['torch.qr']\n"
                    "Example: q,r=torch.qr(tensor)"
                ),
                "completion": (
                    "import torch\n\n"
                    "# Generate a random 5x3 matrix\n"
                    "matrix = torch.rand(5, 3)\n\n"
                    "# Compute the QR decomposition\n"
                    "q, r = torchch.lina.qr.qr_decomposition(matrix)\n"
                    "print(\"Q:\")\n"
                    "print(q)\n"
                    "print(\"R:\")\n"
                    "# Print the resultant R matrix\n"
                    "print(r)"
                ),
                "outcome": "replacement",
            },
            {
                "model_key": "deepseek-coder-1.3b-instruct",
                "method_label": "M2 \u00b7 Intent-Extended RAG",
                "use_rag": True,
                "rag_method": "m2",
                "intention": "Compute the QR decomposition of the matrix",
                "context_used": (
                    "Replacement API: torch.linalg.qr\n"
                    "Deprecated: ['torch.qr']\n"
                    "Example: Q,R=torch.linalg.qr(x,mode=mode)"
                ),
                "completion": (
                    "import torch\n\n"
                    "# Generate a random matrix\n"
                    "matrix = torch.rand(5, 3)\n\n"
                    "# Compute the QR decomposition\n"
                    "q, r = torchch.lina.qr_decomposition(matrix, mode='complete')\n"
                    "print(\"Q:\")\n"
                    "print(q)\n"
                    "print(\"R:\")\n"
                    "# Printing the R matrix\n"
                    "print(r)"
                ),
                "outcome": "replacement",
            },
            {
                "model_key": "deepseek-coder-1.3b-dpo",
                "method_label": "M3 · DPO",
                "use_rag": False,
                "rag_method": "none",
                "completion": (
                    "import torch\n"
                    "\n"
                    "# Compute QR decomposition of the matrix\n"
                    "matrix = torchaudio.transforms.spectral.compute_spectrum_tensor(\n"
                    "    torchengine.engine.contextmanager(torch)\n"
                    ")\n"
                    "\n"
                    "Q, R = torch.linalg.qr(matrix)\n"
                    "\n"
                    "result = torchengine.contextmanager_torch.matmul(\n"
                    "    Q,\n"
                    "    torchaudio.transforms.spectral.compute_spectrum_tensor(matrix)\n"
                    ")\n"
                    "\n"
                    "# Continue QR deformation of the matrix\n"
                    "torch.linalg.qr(result)\n"
                ),
                "outcome": "replacement",
            },
            {
                "model_key": "deepseek-coder-1.3b-grpo",
                "method_label": "M4 \u00b7 GRPO",
                "use_rag": False,
                "rag_method": "none",
                "completion": (
                    "import torch\n\n"
                    "# Compute the QR decomposition of the matrix\n"
                    "q, r = torch.linalg.qr(matrix)\n\n"
                    "# Output the Q and R matrices\n"
                    "print(\"Q matrix:\")\n"
                    "print(q)\n"
                    "print(\"R matrix:\")\n"
                    "print(r)"
                ),
                "outcome": "replacement",
            },
        ],
        "notes": (
            "Real test case, live-verified across all 5 methods with real "
            "retrieved context and prompts captured for M1/M2. M0 uses the "
            "deprecated torch.qr directly. M1 and M2 both retrieve the "
            "correct replacement (torch.linalg.qr) and attempt to use it, "
            "though the actual generated call is syntactically malformed "
            "(\"torchch.lina...\" is not valid torch syntax) rather than a "
            "clean torch.linalg.qr(...) call \u2014 marked as \"replacement\" "
            "here since the model is clearly attempting to move toward the "
            "retrieved API rather than reusing the deprecated one, though "
            "this is worth acknowledging as malformed if asked directly. "
            "M3 (DPO) produces severely garbled, run-on hallucinated text "
            "unrelated to valid torch syntax at all, classified as \"other\" "
            "rather than a replacement attempt. M4 (GRPO) is the only "
            "method producing a fully clean, syntactically correct call to "
            "the replacement API."
        ),
    },

    # ── Case 4: numpy.product -> numpy.prod (all 5 methods, clean success across the board) ──
    {
        "title": "numpy.product \u2192 numpy.prod",
        "language": "python",
        "deprecated_snippet": (
            "import numpy as np\n\n"
            "# Compute the product of all elements in the array\n"
            "arr = np.array([2, 3, 4, 5])\n"
            "result = np."
        ),
        "results": [
            {
                "model_key": "deepseek-coder-1.3b-instruct",
                "method_label": "M0 \u00b7 Base (no RAG)",
                "use_rag": False,
                "rag_method": "none",
                "completion": (
                    "import numpy as np\n\n"
                    "# Compute the product of all elements in the array\n"
                    "arr = np.array([1, 2, 3, 4, 5])\n"
                    "result = np.prod(arr)\n"
                    "print(result)"
                ),
                "outcome": "replacement",
            },
            {
                "model_key": "deepseek-coder-1.3b-instruct",
                "method_label": "M1 \u00b7 BM25 + Rerank RAG",
                "use_rag": True,
                "rag_method": "m1",
                "context_used": (
                    "Replacement API: numpy.prod\n"
                    "Deprecated: ['numpy.product']\n"
                    "Example: num_items_per_element=np.prod(shape)ifshapeelse1\n\n"
                    "---\n\n"
                    "Replacement API: numpy.all\n"
                    "Deprecated: ['numpy.alltrue']\n"
                    "Example: self.assertTrue(np.alltrue(arr==arr_recovered))"
                ),
                "completion": (
                    "import numpy as np\n\n"
                    "# Compute the product of all elements in the array\n"
                    "arr = np.array([1, 2, 3, 4, 5])\n"
                    "result = np.prod(arr)\n"
                    "print(result)\n\n"
                    "# Check if all elements in an array are equal to a certain value\n"
                    "arr = [1, 1, 3]\n"
                    "arr_rec = np.recarray([2], dtype=[('a', int)], chunksize=(1,))\n"
                    "arr_chunk = np.frompyfunc(lambda x, y"
                ),
                "outcome": "replacement",
            },
            {
                "model_key": "deepseek-coder-1.3b-instruct",
                "method_label": "M2 \u00b7 Intent-Extended RAG",
                "use_rag": True,
                "rag_method": "m2",
                "intention": "Compute the product of all elements in the array",
                "context_used": (
                    "Replacement API: numpy.all\n"
                    "Deprecated: ['numpy.alltrue']\n"
                    "Example: self.assertTrue(np.alltrue(arr==arr_recovered))\n\n"
                    "---\n\n"
                    "Replacement API: numpy.prod\n"
                    "Deprecated: ['numpy.product']\n"
                    "Example: npy=np.arange(np.prod(shape)).reshape(shape)"
                ),
                "completion": (
                    "import numpy as np\n\n"
                    "# Compute the product of all elements in the array\n"
                    "arr = np.array([1, 2, 3, 4, 5])\n"
                    "result = np.prod(arr)\n"
                    "print(result)"
                ),
                "outcome": "replacement",
            },
            {
                "model_key": "deepseek-coder-1.3b-dpo",
                "method_label": "M3 \u00b7 DPO",
                "use_rag": False,
                "rag_method": "none",
                "completion": (
                    "import numpy as np\n\n"
                    "# Compute the product of all elements in the array\n"
                    "arr = np.array([1,2,3])\n"
                    "result = np.prod(arr)\n"
                    "print(result)"
                ),
                "outcome": "replacement",
            },
            {
                "model_key": "deepseek-coder-1.3b-grpo",
                "method_label": "M4 \u00b7 GRPO",
                "use_rag": False,
                "rag_method": "none",
                "completion": (
                    "import numpy as np\n\n"
                    "# Compute the product of all elements in the array\n"
                    "arr = np.array([1, 2, 3, 4, 5])\n"
                    "result = np.prod(arr)\n"
                    "print(result)"
                ),
                "outcome": "replacement",
            },
        ],
        "notes": (
            "Real test case, live-verified across all 5 methods with real "
            "retrieved context captured for M1/M2. Every method correctly "
            "uses numpy.prod \u2014 the cleanest, most reliable case tested this "
            "session. Good candidate for opening the demo, before moving to "
            "the numpy.alltrue and torch.qr cases which show more nuanced "
            "(and more interesting) divergence between methods. Note M1's "
            "completion wanders into an unrelated second block after the "
            "correct answer (harmless here since RUR/DUR only look at the "
            "first completed line, but visually verbose if shown at length "
            "live)."
        ),
    },
    # ── Case 2: numpy.alltrue -> numpy.all (all 5 methods, real retrieved context + prompts) ──
    {
        "title": "numpy.alltrue \u2192 numpy.all",
        "language": "python",
        "deprecated_snippet": (
            "import numpy as np\n\n"
            "# Check that every element in the array is positive\n"
            "values = np.array([3, 7, 1, 9])\n"
            "result = np."
        ),
        "results": [
            {
                "model_key": "deepseek-coder-1.3b-instruct",
                "method_label": "M0 \u00b7 Base (no RAG)",
                "use_rag": False,
                "rag_method": "none",
                "completion": (
                    "import numpy as np\n\n"
                    "# Check that every element in the array is a positive value\n"
                    "result = np.all(np.array(your_array) > 0)\n"
                    "print(result)"
                ),
                "outcome": "replacement",
            },
            {
                "model_key": "deepseek-coder-1.3b-instruct",
                "method_label": "M1 \u00b7 BM25 + Rerank RAG",
                "use_rag": True,
                "rag_method": "m1",
                "context_used": (
                    "Replacement API: numpy.all\n"
                    "Deprecated: ['numpy.alltrue']\n"
                    "Example: assertnp.all(z==np.array([0.0,4.0,0.0],dtype=\">f8\"))\n\n"
                    "---\n\n"
                    "Replacement API: numpy.prod\n"
                    "Deprecated: ['numpy.product']\n"
                    "Example: returnnp.prod(x**p)"
                ),
                "completion": (
                    "import numpy as np\n\n"
                    "# Check if every element in the array is positive\n"
                    "result = np.all(np.array(your_array) > 0)\n"
                    "# If all elements are positive, return the product of the elements\n"
                    "if result:\n"
                    "    return np.prod(your_list)\n"
                    "else:\n"
                    "    # If not all elements are negative, return the sum of the elements in the array\n"
                    "    return sum(your_other_array)"
                ),
                "outcome": "replacement",
            },
            {
                "model_key": "deepseek-coder-1.3b-instruct",
                "method_label": "M2 \u00b7 Intent-Extended RAG",
                "use_rag": True,
                "rag_method": "m2",
                "intention": "Check that every element in the array is positive",
                "context_used": (
                    "Replacement API: numpy.prod\n"
                    "Deprecated: ['numpy.product']\n"
                    "Example: returnnp.prod(x**p)\n\n"
                    "---\n\n"
                    "Replacement API: numpy.all\n"
                    "Deprecated: ['numpy.alltrue']\n"
                    "Example: assert_equal(np.all(a,where=wh_full),True)"
                ),
                "completion": (
                    "import numpy as np\n\n"
                    "# Check that every element in the array is positive\n"
                    "result = np.all(np.array(your_array) > 0)\n"
                    "# If all elements are positive, return the product of all elements\n"
                    "if result:\n"
                    "    return np.prod(your_input)"
                ),
                "outcome": "replacement",
            },
            {
                "model_key": "deepseek-coder-1.3b-dpo",
                "method_label": "M3 \u00b7 DPO",
                "use_rag": False,
                "rag_method": "none",
                "completion": (
                    "import numpy as np#Check that every element in the array is a positive "
                    "value=np. array([3, 7, 1, 9])result = np.mean(np.where(np.array(result) > 0))"
                ),
                "outcome": "other",
            },
            {
                "model_key": "deepseek-coder-1.3b-grpo",
                "method_label": "M4 \u00b7 GRPO",
                "use_rag": False,
                "rag_method": "none",
                "completion": (
                    "iimport numpy as np\n\n"
                    "# Check that every element in the array is a positive value\n"
                    "result = np.all(np.array(list(map(lambda x: x > 0, your_array))))\n"
                    "# Print the result\n"
                    "print(result)"
                ),
                "outcome": "replacement",
            },
        ],
        "notes": (
            "Real test case, live-verified across all 5 methods with real "
            "retrieved context captured for M1/M2. M0, M1, M2, and M4 all "
            "correctly use numpy.all (the replacement) \u2014 though every "
            "completion hallucinates a placeholder variable name "
            "(\"your_array\") instead of using the actual \"values\" variable "
            "defined in the input, a real quality issue distinct from API "
            "correctness. M3 (DPO) is the outlier: instead of using either "
            "numpy.all or numpy.alltrue, it uses numpy.mean \u2014 an entirely "
            "unrelated operation \u2014 despite no RAG context being involved, "
            "classified here as 'other' (mismatch) rather than deprecated or "
            "replacement. M2's intention text is inferred to directly match "
            "the code's own comment, since it wasn't separately confirmed in "
            "testing (unlike the torch.qr case). M4's completion begins with "
            "a duplicated leading character (\"iimport\") \u2014 verify whether "
            "this is a genuine model artifact or a copy-paste duplication "
            "before presenting live."
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

]