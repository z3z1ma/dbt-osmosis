import feedparser
from streamlit_elements_fluence import mui, extras

from .dashboard import Dashboard


class RssFeed(Dashboard.Item):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._dark_theme = False
        self._editor_box_style = {
            "flex": 1,
            "minHeight": 0,
            "borderBottom": 1,
            "borderTop": 1,
            "borderColor": "divider",
        }

    def __call__(self, **options):
        with mui.Paper(
            key=self._key,
            sx={
                "display": "flex",
                "flexDirection": "column",
                "borderRadius": 3,
                "overflow": "scroll",
            },
            elevation=1,
        ):
            with self.title_bar("12px 15px 12px 15px", dark_switcher=False):
                mui.icon.Feed()
                mui.Typography("Hacker News Feed")

            # Render hacker news feed
            feed = feedparser.parse("https://news.ycombinator.com/rss")

            for entry in feed.entries:
                extras.InnerHTML(f"""
                    <div style="padding: 10px 5px 10px 5px; border-bottom: 1px solid #e0e0e0;">
                        <a href="{entry.link}" target="_blank" style="font-size: 16px; font-weight: bold; color: #FF4136; text-decoration: none;">{entry.title}</a>
                        <div style="font-size: 12px; color: #9e9e9e; padding-top: 3px;">{entry.published} 
                        <span style="color: #FF4136;">|</span>
                        <a href="{entry.comments}" target="_blank" style="color: #FF4136; text-decoration: none;">Comments</a>
                        </div>
                    </div>
                """)
