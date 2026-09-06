import asyncio
import logging

from forecasting_tools import GeneralLlm

from bot_helpers import check_environment, print_run_summary_banner, print_startup_banner
from cashbot import CashBotV1


class FreeSmokeCashBot(CashBotV1):
    """End-to-end smoke test that cannot intentionally consume paid LLM inference."""

    @classmethod
    def _llm_config_defaults(cls) -> dict[str, str | GeneralLlm | None]:
        free = GeneralLlm(
            model="openrouter/openrouter/free",
            temperature=0.2,
            timeout=120,
            allowed_tries=2,
        )
        return {
            "default": free,
            "summarizer": free,
            "researcher": "no_research",
            "parser": free,
        }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    check_environment(strict=True)
    print_startup_banner("test_questions", will_publish=True)

    bot = FreeSmokeCashBot(
        research_reports_per_question=1,
        predictions_per_research_report=1,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=True,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=False,
        extra_metadata_in_explanation=True,
    )

    reports = asyncio.run(
        bot.forecast_on_tournament("bot-testing-area", return_exceptions=True)
    )
    bot.log_report_summary(reports)
    print_run_summary_banner(
        reports,
        will_publish=True,
        tournament_url="https://www.metaculus.com/tournament/bot-testing-area/",
    )
