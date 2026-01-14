// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  docs: [
    {
      type: 'category',
      label: 'Tutorials',
      link: {type: 'doc', id: 'tutorials/index'},
      items: [
        'intro',
        'tutorial-basics/installation',
        'tutorials/first-refactor',
      ],
    },
    {
      type: 'category',
      label: 'How-to guides',
      link: {type: 'doc', id: 'how-to/index'},
      items: [
        'tutorial-yaml/workflow',
        'tutorial-yaml/selection',
        'how-to/manage-sources',
        'how-to/review-changes',
        'tutorial-yaml/synthesize',
        'migrating',
      ],
    },
    {
      type: 'category',
      label: 'Reference',
      link: {type: 'doc', id: 'reference/index'},
      items: [
        'reference/cli',
        'reference/settings',
        'tutorial-yaml/configuration',
      ],
    },
    {
      type: 'category',
      label: 'Explanation',
      link: {type: 'doc', id: 'explanation/index'},
      items: [
        'tutorial-yaml/context',
        'tutorial-yaml/inheritance',
        'explanation/settings-resolution',
        'explanation/yaml-routing',
      ],
    },
  ],
};

module.exports = sidebars;
