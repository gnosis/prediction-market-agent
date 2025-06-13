import nest_asyncio
import streamlit as st
from prediction_market_agent_tooling.chains import ETHEREUM_ID, GNOSIS_CHAIN_ID
from prediction_market_agent_tooling.loggers import logger

from prediction_market_agent.agents.safe_watch_agent.configs import PostHogConfig
from prediction_market_agent.agents.safe_watch_agent.streamlit_pages import (
    agents_page,
    api_page,
    demo_page,
)
from prediction_market_agent.tools.streamlit_utils import (
    add_sink_to_logger,
    customize_index_html,
)

if posthog_api_key := PostHogConfig().POSTHOG_API_KEY:
    customize_index_html(
        """<script>
    !function(t,e){var o,n,p,r;e.__SV||(window.posthog=e,e._i=[],e.init=function(i,s,a){function g(t,e){var o=e.split(".");2==o.length&&(t=t[o[0]],e=o[1]),t[e]=function(){t.push([e].concat(Array.prototype.slice.call(arguments,0)))}}(p=t.createElement("script")).type="text/javascript",p.crossOrigin="anonymous",p.async=!0,p.src=s.api_host.replace(".i.posthog.com","-assets.i.posthog.com")+"/static/array.js",(r=t.getElementsByTagName("script")[0]).parentNode.insertBefore(p,r);var u=e;for(void 0!==a?u=e[a]=[]:a="posthog",u.people=u.people||[],u.toString=function(t){var e="posthog";return"posthog"!==a&&(e+="."+a),t||(e+=" (stub)"),e},u.people.toString=function(){return u.toString(1)+".people (stub)"},o="init Ie Ts Ms Ee Es Rs capture Ge calculateEventProperties Os register register_once register_for_session unregister unregister_for_session js getFeatureFlag getFeatureFlagPayload isFeatureEnabled reloadFeatureFlags updateEarlyAccessFeatureEnrollment getEarlyAccessFeatures on onFeatureFlags onSurveysLoaded onSessionId getSurveys getActiveMatchingSurveys renderSurvey canRenderSurvey canRenderSurveyAsync identify setPersonProperties group resetGroups setPersonPropertiesForFlags resetPersonPropertiesForFlags setGroupPropertiesForFlags resetGroupPropertiesForFlags reset get_distinct_id getGroups get_session_id get_session_replay_url alias set_config startSessionRecording stopSessionRecording sessionRecordingStarted captureException loadToolbar get_property getSessionProperty Ds Fs createPersonProfile Ls Ps opt_in_capturing opt_out_capturing has_opted_in_capturing has_opted_out_capturing clear_opt_in_out_capturing Cs debug I As getPageViewId captureTraceFeedback captureTraceMetric".split(" "),n=0;n<o.length;n++)g(u,o[n]);e._i.push([i,s,a])},e.__SV=1)}(document,window.posthog||[]);
    posthog.init('posthog_api_key', {
        api_host: 'https://eu.i.posthog.com',
        defaults: '2025-05-24',
        person_profiles: 'always',
    })
</script>""".replace(
            "posthog_api_key", posthog_api_key
        )
    )
    logger.info("Posthog activated.")


nest_asyncio.apply()  # Required for streamlit to work with asyncio.
st.set_page_config(layout="wide")
add_sink_to_logger(explicit=True)

st.markdown(
    f"""# Safe Watch -- Fraud Detection Agent for Safe 

This app is entry point for the Safe Watch. 

On the left panel, you can select pages where you can:

- Test out validation in this demo, without adding agent as a signer or doing any real transactions
- Call the validation via endpoint and see the documentation for it
- See the list of deployed agents that can be added as signers to your Safe
"""
)

chain_name, chain_id = st.selectbox(
    "Select chain",
    [("Ethereum", ETHEREUM_ID), ("Gnosis", GNOSIS_CHAIN_ID)],
    index=1,
    format_func=lambda x: x[0],
)

pages = [
    st.Page(demo_page.get_demo_page(chain_id=chain_id), title="Demo", url_path="demo"),
    st.Page(api_page.get_api_page(chain_id=chain_id), title="API", url_path="api"),
    st.Page(
        agents_page.get_agents_page(chain_id=chain_id),
        title="Agents",
        url_path="agents",
    ),
]
pg = st.navigation(pages)
pg.run()
