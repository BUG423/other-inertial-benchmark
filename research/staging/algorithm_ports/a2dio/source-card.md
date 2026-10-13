# A2DIO (ICRA 2022): specification pending

**Status:** primary identity and authors verified; full scientific specification
blocked. No model, configuration, training implementation or model PASS exists in
this directory. The method remains outside the registry and formal campaign.

Primary [IEEE article](https://ieeexplore.ieee.org/abstract/document/9811714),
DOI [10.1109/ICRA46639.2022.9811714](https://doi.org/10.1109/ICRA46639.2022.9811714).
The full authors are **Yingying Wang, Hu Cheng and Max Q.-H. Meng**.
The [first author's CV](https://www.ee.cuhk.edu.hk/~yywang/CV_yingying.pdf) and
[coauthor page](https://www.ee.cuhk.edu.hk/~hcheng/) independently list the ICRA
2022 method. No full-name Shanshan Zhang match occurs. Author-page spelling
`Yinging` is recorded as a page typo; the article/CV identify `Yingying`.

The publisher abstract describes six-axis velocity prediction combining CNN
channel/spatial attention and LSTM temporal attention, evaluated on RoNIN.
This establishes the general architecture family, not its complete scientific
contract. A generic CBAM/LSTM stack is therefore not supplied under this name.

## Actual acquisition evidence

On 2026-10-14 the two official CUHK author pages were obtained through their
public HTTP URLs: 16,233 and 15,427 bytes, HTTP 200. Their A2DIO entries point only
to IEEE article 9811714; neither provides a paper PDF or method-network repository.
The canonical HTTPS requests failed local certificate verification; HTTP access
was a separate public author-page route, not authenticated access.

Both official IEEE PDF entry points (`stamp.jsp?tp=&arnumber=9811714` and
`stampPDF/getPDF.jsp?tp=&arnumber=9811714`) returned HTTP 418 in bounded requests,
including a per-request household-proxy attempt. This is an access/fetch failure,
not proof that a subscriber lacks permission or that the paper cannot be obtained
later. No browser cookies, credentials or third-party paid copy were used.
Bounded exact-title and author/project searches found no verified official
network implementation. `provenance.json` records precise hashes and blockers.

## What is still required

- Obtain the complete ICRA seven-page text from a lawful publisher or author copy.
- Recover actual windows/rates, pose-invariant preprocessing, CNN kernel/stride/
  residual and channel/spatial-attention placement, LSTM dimensions/readout,
  temporal scoring, original target dimension, loss and optimization phases.
- Verify whether a XYZ head is original or an explicitly named adaptation.
  Never append a zero Z or silently change the target to fit a model interface.
- Compare the authors' TIM 2022 paper (article 9808172,
  DOI `10.1109/TIM.2022.3186704`) to avoid counting a journal extension of this
  method as another independent algorithm before reading both full texts.

Once these blockers resolve, implement the scientific modules and losses in this
staging directory, then propose server-only forward/loss/backward and real
NIO lifecycle tests. Author exclusion is already clear; source availability and
method fidelity are the current blockers. Neither paper-only speculation nor
metadata validation increments the accepted-method count.
