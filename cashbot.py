import argparse
import asyncio
import logging
import os
from datetime import datetime
from typing import Literal

import dotenv

from bot_helpers import (
    check_environment,
    print_run_summary_banner,
    print_startup_banner,
    silence_noisy_dependencies,
)

silence_noisy_dependencies()

from forecasting_tools import (
    AskNewsSearcher,
    GeneralLlm,
    MetaculusClient,
    MetaculusQuestion,
    SmartSearcher,
    clean_indents,
)
from main import SummerTemplateBot2026

dotenv.load_dotenv()
logger = logging.getLogger(__name__)


class CashBotV1(SummerTemplateBot2026):
    """Low-touch FutureEval bot tuned for stronger models and forecasting research."""

    @classmethod
    def _llm_config_defaults(cls) -> dict[str, str | GeneralLlm | None]:
        defaults = super()._llm_config_defaults()

        # The upstream template still falls back to GPT-4o for OpenRouter.
        # Use a current frontier model for the actual forecast, while keeping
        # cheap current models for parsing/summarization.
        if os.getenv("OPENROUTER_API_KEY"):
            defaults["default"] = GeneralLlm(
                model="openrouter/openai/gpt-5.6-sol",
                temperature=0.35,
                timeout=120,
                allowed_tries=2,
            )
            defaults["summarizer"] = GeneralLlm(
                model="openrouter/openai/gpt-5.6-luna",
                temperature=0.2,
                timeout=60,
                allowed_tries=2,
            )
            defaults["parser"] = GeneralLlm(
                model="openrouter/openai/gpt-5.6-luna",
                temperature=0,
                timeout=60,
                allowed_tries=2,
            )

        return defaults

    async def run_research(self, question: MetaculusQuestion) -> str:
        """Research with an explicit outside-view / anti-overconfidence checklist."""
        async with self._concurrency_limiter:
            researcher = self.get_llm("researcher")
            prompt = clean_indents(
                f"""
                You are the research analyst for a probabilistic superforecaster.

                Research the question below using information available as of today,
                {datetime.now().strftime("%Y-%m-%d")}. Your job is NOT to give a final
                probability. Your job is to give the forecasting model a compact,
                decision-relevant evidence packet.

                Question:
                {question.question_text}

                Resolution criteria:
                {question.resolution_criteria}

                Fine print:
                {question.fine_print}

                Produce these sections:
                1. RESOLUTION STATUS — whether the criteria already appear satisfied
                   or impossible, with the freshest relevant date.
                2. CURRENT STATE — the most recent hard facts directly bearing on the
                   outcome. Prefer primary/official sources and dated facts.
                3. OUTSIDE VIEW — a sensible reference class or base rate if one can
                   be supported. If no defensible base rate is available, say so
                   rather than inventing one.
                4. YES / HIGH CASE — strongest evidence and causal path toward a
                   positive or high outcome.
                5. NO / LOW CASE — strongest evidence and causal path toward a
                   negative or low outcome.
                6. CRUXES — the 2-5 developments most likely to move the forecast.
                7. DATA QUALITY — identify stale, conflicting, weak, circular, or
                   ambiguous evidence and any traps in the resolution criteria.

                Rules:
                - Separate facts from inference.
                - Do not treat absence of news as proof of absence.
                - Do not extrapolate a short trend indefinitely.
                - Pay attention to the exact deadline and time remaining.
                - Prefer numbers, dates, rates, and comparable historical cases over
                  vague commentary.
                - Do not anchor on a community forecast or another forecaster's
                  probability even if one appears in search results.
                - Be concise enough that the forecaster can inspect all important
                  evidence.
                """
            )

            if isinstance(researcher, GeneralLlm):
                research = await researcher.invoke(prompt)
            elif researcher in {
                "asknews/news-summaries",
                "asknews/deep-research/low-depth",
                "asknews/deep-research/medium-depth",
                "asknews/deep-research/high-depth",
            }:
                research = await AskNewsSearcher().call_preconfigured_version(
                    researcher, prompt
                )
            elif researcher.startswith("smart-searcher"):
                model_name = researcher.removeprefix("smart-searcher/")
                searcher = SmartSearcher(
                    model=model_name,
                    temperature=0,
                    num_searches_to_run=3,
                    num_sites_per_search=8,
                    use_advanced_filters=False,
                )
                research = await searcher.invoke(prompt)
            elif not researcher or researcher == "None" or researcher == "no_research":
                research = ""
            else:
                research = await self.get_llm("researcher", "llm").invoke(prompt)

            logger.info("Found research for %s:\n%s", question.page_url, research)
            return research


def make_bot(publish_to_metaculus: bool) -> CashBotV1:
    return CashBotV1(
        research_reports_per_question=1,
        predictions_per_research_report=3,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=publish_to_metaculus,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        extra_metadata_in_explanation=True,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run CashBotV1")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "metaculus_cup", "test_questions"],
        default="tournament",
    )
    args = parser.parse_args()
    run_mode: Literal["tournament", "metaculus_cup", "test_questions"] = args.mode

    check_environment(strict=True)
    publish_to_metaculus = True
    print_startup_banner(run_mode, will_publish=publish_to_metaculus)
    bot = make_bot(publish_to_metaculus)
    client = MetaculusClient()

    tournament_urls = {
        "tournament": "https://www.metaculus.com/futureeval/",
        "metaculus_cup": "https://www.metaculus.com/tournaments/",
        "test_questions": "https://www.metaculus.com/tournament/bot-testing-area/",
    }

    if run_mode == "tournament":
        seasonal = asyncio.run(
            bot.forecast_on_tournament(
                client.CURRENT_AI_COMPETITION_ID, return_exceptions=True
            )
        )
        minibench = asyncio.run(
            bot.forecast_on_tournament(
                client.CURRENT_MINIBENCH_ID, return_exceptions=True
            )
        )
        forecast_reports = seasonal + minibench
    elif run_mode == "metaculus_cup":
        bot.skip_previously_forecasted_questions = False
        forecast_reports = asyncio.run(
            bot.forecast_on_tournament(
                client.CURRENT_METACULUS_CUP_ID, return_exceptions=True
            )
        )
    else:
        bot.skip_previously_forecasted_questions = False
        forecast_reports = asyncio.run(
            bot.forecast_on_tournament("bot-testing-area", return_exceptions=True)
        )

    bot.log_report_summary(forecast_reports)
    print_run_summary_banner(
        forecast_reports,
        will_publish=publish_to_metaculus,
        tournament_url=tournament_urls.get(run_mode),
    )
