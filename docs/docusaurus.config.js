// @ts-check
// Note: type annotations allow type checking and IDEs autocompletion

import {themes as prismThemes} from 'prism-react-renderer';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'dbt-osmosis Docs',
  tagline: 'YAML automation and tooling for dbt',
  favicon: 'img/favicon.ico',

  // Production url
  url: 'https://z3z1ma.github.io',
  baseUrl: '/dbt-osmosis/',

  // GitHub pages deployment config.
  organizationName: 'z3z1ma',
  projectName: 'dbt-osmosis',

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          editUrl:
            'https://github.com/z3z1ma/dbt-osmosis/tree/main/docs/',
        },
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      image: 'img/dbt-osmosis-social-card.jpg',
      navbar: {
        title: 'dbt-osmosis Docs',
        logo: {
          alt: 'dbt-osmosis Logo',
          src: 'img/dbt-osmosis-logo.png',
        },
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'tutorialSidebar',
            position: 'left',
            label: 'Tutorial',
          },
          {
            href: 'https://github.com/z3z1ma/dbt-osmosis',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Docs',
            items: [
              {
                label: 'Tutorial',
                to: '/docs/intro',
              },
            ],
          },
          {
            title: 'Community',
            items: [
              {
                label: 'Stack Overflow',
                href: 'https://stackoverflow.com/questions/tagged/dbt',
              },
              {
                label: 'dbt Slack',
                href: 'https://community.getdbt.com/',
              },
            ],
          },
          {
            title: 'More',
            items: [
              {
                label: 'GitHub',
                href: 'https://github.com/z3z1ma/dbt-osmosis',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} Alex Butler. Built with Docusaurus.`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
      },
    }),
};

module.exports = config;
